# Take in a pgn file
# Output the binary map of each board after each move in the game

# TODO: save dataset files to HDF5 file
# TODO: decaying result label + elo trust


import chess
import chess.pgn
import numpy as np
import os
import io

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
        # flip board to always see current player's perspective
        if board.turn == chess.BLACK:
            self.board = board.transform(chess.flip_vertical)
        else:
            self.board = board

        # 1 for win, 0 for draw, -1 for loss, n
        # NOT SURE IF THIS IS THE BEST WAY TO DO THIS
        if result == '1-0' and  self.board.turn == chess.WHITE:
            self.result = 1
        elif result == '0-1' and  self.board.turn == chess.BLACK:
            self.result = 1
        elif result == '1/2-1/2':
            self.result = 0
        else:
            self.result = -1

        self.board_map = np.zeros((8, 8, 19), dtype=np.uint8)
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

        self.board_map[:,:,18] = self.result # label to be predicted

        return self.board_map

def get_boards(pgn_files):
    boards = []
    for pgn in pgn_files:

        # handle multiple games in one pgn file
        games = []
        pgn_read = pgn.read()
        lines = pgn_read.split('\n\n')
        for i in range(len(lines)//2):
            games.append(lines[2*i] + '\n\n' + lines[2*i+1] + '\n\n')
            if i % 10000 == 0:
                print("formatting game", i)

        print("finished formatting games for file", pgn.name)
            
        # for debugging
        # for cur_game in [games[0]]:
        counter = 0 

        for cur_game in games:
            counter += 1
            if counter % 1000 == 0:
                print("processing game", counter)
            game = chess.pgn.read_game(io.StringIO(cur_game))
            # print(cur_game)
            if game.headers['Result'] == '1-0' or game.headers['Result'] == '0-1' or game.headers['Result'] == '1/2-1/2':
                if int(game.headers['WhiteElo']) > 1500:
                    board = game.board()
                    i = 0
                    for move in game.mainline_moves():
                        board.push(move)
                        boards.append(Board(board, game.headers['Result']))
                        # save image of board, extremely slow!!
                        svg = chess.svg.board(board=board)
                        with open('utils/trash/'+'board' + str(i) + '.svg', 'w+') as f:
                            f.write(svg)
                        i += 1
    return boards


if __name__ == '__main__':
    boards = get_boards(get_pgn_files(PATH))
    print(len(boards))
    print(boards[1].board_map[:,:,0])
    
