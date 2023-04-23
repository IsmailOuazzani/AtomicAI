from typing import Dict, List, Any
import chess
import sys
import time
from evaluate import check_end_game, move_value

debug_info: Dict[str, Any] = {}


MATE_SCORE     = 1000000000
MATE_THRESHOLD =  999000000


load_start = time.time()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




import torch.nn.functional as F
class ChessNet2(nn.Module):
    def __init__(self):
        super(ChessNet2, self).__init__()
        self.name = "ChessNet2"
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=17, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=256 * 2 * 2, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.fc_layers(x)
        x = torch.sigmoid(x)
        return x



class Board():
    def __init__(self, board, result):
        # 1 for win, 0 for draw, -1 for loss, n
        if result == '1-0' and  board.turn == chess.WHITE:
            self.result = 1
        elif result == '0-1' and  board.turn == chess.BLACK:
            self.result = 1
        elif result == '1/2-1/2':
            self.result = 0
        else:
            self.result = -1

        self.board = board

        self.board_map = np.zeros((8, 8, 18))
        self.board_map = self.get_board_map()

    def get_board_map(self):
        # Piece map
        for i in range(8):
            for j in range(8):
                piece = self.board.piece_at(chess.square(i, j))
                if piece :
                    if piece.color:
                        self.board_map[i,j,piece.piece_type-1] = 1
                    else:
                        self.board_map[i,j,piece.piece_type + 5] = 1

        """
        Have not tested the following code yet
        """
        # castling rights
        if self.board.has_kingside_castling_rights(chess.WHITE):
            self.board_map[:,:,12] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            self.board_map[:,:,13] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            self.board_map[:,:,14] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            self.board_map[:,:,15] = 1
        # en passant
        if self.board.ep_square: 
            ep = chess.square_file(self.board.ep_square)
            self.board_map[:,ep,16] = 1

        self.board_map[:,:,17] = self.result # label to be predicted
        return self.board_map.transpose((2,0,1))

################# Load desired model here #################
model_path = 'models/'
NN = 'model_ChessNet2_bs1000_lr0.001_epoch49'
# NN = 'model_ChessNet2_bs10000_lr0.001_epoch30'
# NN = 'model_ChessNet2_bs10000_lr0.001_epoch99'
NN = model_path + NN
# load the model
model = ChessNet2()
state_dict = torch.load(NN, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()
print('Model loaded in {} seconds'.format(time.time() - load_start))
   

def evaluate_board(board: chess.Board) -> float:
    """
    Evaluates the full board and determines which player is in a most favorable position.
    The sign indicates the side:
        (+) for white
        (-) for black
    The magnitude, how big of an advantage that player has
    """

    total = 0
    end_game = check_end_game(board)

    # convert the board to a tensor
    b = Board(board, '1-0')
    board_map = b.board_map
    board_map = torch.from_numpy(board_map[:17,:,:]).float()
    board_map = board_map.unsqueeze(0)

    # evaluate the board
    with torch.no_grad():
        total = model(board_map)
        # transform to float
        total = total.item()

    total = total if board.turn == chess.WHITE else -total

    return total


def next_move_chessnet2(depth: int, board: chess.Board, debug=True) -> chess.Move:
    """
    What is the next best move?
    """
    debug_info.clear()
    debug_info["nodes"] = 0
    t0 = time.time()

    move = minimax_root(depth, board)
    best_board = board.copy(stack=False)
    best_board.push(move)
    eval_time = time.time()
    eval = evaluate_board(best_board)
    print(f"Eval time: {time.time() - eval_time} seconds")

    debug_info["time"] = time.time() - t0
    if debug == True:
        print(f"info {debug_info}")
    return move, eval


def get_ordered_moves(board: chess.Board) -> List[chess.Move]:
    """
    Get legal moves.
    Attempt to sort moves by best to worst.
    Use piece values (and positional gains/losses) to weight captures.
    """
    end_game = check_end_game(board)

    def orderer(move):
        board.push(move)
        value = evaluate_board(board)
        board.pop()
        return value

    in_order = sorted(
        board.legal_moves, key=orderer, reverse=(board.turn == chess.WHITE)
    )
    return list(in_order)


def minimax_root(depth: int, board: chess.Board) -> chess.Move:
    """
    What is the highest value move per our evaluation function?
    """
    # White always wants to maximize (and black to minimize)
    # the board score according to evaluate_board()
    maximize = board.turn == chess.WHITE
    best_move = -float("inf")
    if not maximize:
        best_move = float("inf")

    moves = get_ordered_moves(board)
    best_move_found = moves[0]

    for move in moves:
        board.push(move)
        # Checking if draw can be claimed at this level, because the threefold repetition check
        # can be expensive. This should help the bot avoid a draw if it's not favorable
        # https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.can_claim_draw
        if board.can_claim_draw():
            value = 0.0
        else:
            value = minimax(depth - 1, board, -float("inf"), float("inf"), not maximize)
        board.pop()
        if maximize and value >= best_move:
            best_move = value
            best_move_found = move
        elif not maximize and value <= best_move:
            best_move = value
            best_move_found = move

    return best_move_found


def minimax(
    depth: int,
    board: chess.Board,
    alpha: float,
    beta: float,
    is_maximising_player: bool,
) -> float:
    """
    Core minimax logic.
    https://en.wikipedia.org/wiki/Minimax
    """
    debug_info["nodes"] += 1

    if board.is_checkmate() or board.is_variant_end():
        # The previous move resulted in checkmate
        return -MATE_SCORE if is_maximising_player else MATE_SCORE
    # When the game is over and it's not a checkmate it's a draw
    # In this case, don't evaluate. Just return a neutral result: zero
    elif board.is_game_over():
        return  -MATE_SCORE if is_maximising_player else MATE_SCORE

    if depth == 0:
        return evaluate_board(board)

    if is_maximising_player:
        best_move = -float("inf")
        moves = get_ordered_moves(board)
        for move in moves:
            board.push(move)
            curr_move = minimax(depth - 1, board, alpha, beta, not is_maximising_player)
            # Each ply after a checkmate is slower, so they get ranked slightly less
            # We want the fastest mate!
            if curr_move > MATE_THRESHOLD:
                curr_move -= 1
            elif curr_move < -MATE_THRESHOLD:
                curr_move += 1
            best_move = max(
                best_move,
                curr_move,
            )
            board.pop()
            alpha = max(alpha, best_move)
            if beta <= alpha:
                return best_move
        return best_move
    else:
        best_move = float("inf")
        moves = get_ordered_moves(board)
        for move in moves:
            board.push(move)
            curr_move = minimax(depth - 1, board, alpha, beta, not is_maximising_player)
            if curr_move > MATE_THRESHOLD:
                curr_move -= 1
            elif curr_move < -MATE_THRESHOLD:
                curr_move += 1
            best_move = min(
                best_move,
                curr_move,
            )
            board.pop()
            beta = min(beta, best_move)
            if beta <= alpha:
                return best_move
        return best_move
