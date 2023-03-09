# Take in a pgn file
# Output the binary map of each board after each move in the game

import chess
import chess.pgn
import numpy as np
import os

# games are located in the dataset folder in the parent directory in the form of pgn files
PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset\\')

# read pgns
def get_pgn_files(path):
    pgn_files = []
    for file in os.listdir(path):
        if file.endswith('.pgn'):
            pgn_files.append(open(path + file, 'r'))

    return pgn_files

class Board():
    def __init__(self, board, result):
        self.board = board
        self.result = result
        self.board_map = np.zeros((8, 8, 18), dtype=np.uint8)
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
        # color
        if self.board.turn:
            self.board_map[:,:,12] = 1


        """
        Have not tested the following code yet
        """
        # castling rights
        if self.board.has_kingside_castling_rights(chess.WHITE):
            self.board_map[:,:,13] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            self.board_map[:,:,14] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            self.board_map[:,:,15] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            self.board_map[:,:,16] = 1
        # en passant
        if self.board.ep_square: 
            ep = chess.square_file(self.board.ep_square)
            self.board_map[:,ep,17] = 1


        return self.board_map

def get_boards(pgn_files):
    boards = []
    for pgn in pgn_files:

        game = chess.pgn.read_game(pgn)
        board = game.board()
        # boards.append(Board(board)) # uncomment this line to get the board before the first move
        i = 0
        for move in game.mainline_moves():
            board.push(move)
            boards.append(Board(board, game.headers['Result']))
            # # save image of board
            # svg = chess.svg.board(board=board)
            # with open('board' + str(i) + '.svg', 'w+') as f:
            #     f.write(svg)
            # i += 1
    return boards

if __name__ == '__main__':
    boards = get_boards(get_pgn_files(PATH))
    print(boards[0].get_board_map()[:,:,0])
    print(boards[0].result)