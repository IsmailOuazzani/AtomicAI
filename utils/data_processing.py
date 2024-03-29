# Take in a pgn file
# Output the binary map of each board after each move in the game

# TODO: decaying result label + elo trust

import chess
import chess.pgn
import numpy as np
import os
import io
import sys
import random

GAME_PER_FILE = 5000 # 10 000 is too slow

# games are located in the dataset folder in the parent directory in the form of pgn files
# games need to be uncompressed first (see README.md)
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
        # 1 for win, 0 for draw, -1 for loss, n
        # NOT SURE IF THIS IS THE BEST WAY TO DO THIS
        if result == '1-0' and  board.turn == chess.WHITE:
            self.result = 1
        elif result == '0-1' and  board.turn == chess.BLACK:
            self.result = 1
        elif result == '1/2-1/2':
            self.result = 0
        else:
            self.result = 0

        # flip board to always see current player's perspective
        if board.turn == chess.BLACK: # since we don't care about game history, mirror the board
            # self.board = board.transform(chess.flip_vertical) ## flips board
            self.board = board.mirror() ## flips board and colors
        else:
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

        # flip board to always put the current player as white
        return self.board_map.transpose((2,0,1))

def get_boards(pgn_files):
    # get the board from all the files, given a maximum number of boards and an elo threshold
    boards = []
    data_counter = 0
    file_counter = 0
    for pgn in pgn_files:
        # handle multiple games in one pgn file
        games = []
        pgn_read = pgn.read()
        # replace \n\n\n with \n\n
        pgn_clean = pgn_read.replace('\n\n\n', '\n\n')
        pgn_clean = pgn_clean.replace('\n\n\n\n', '\n\n')
        lines = pgn_clean.split('\n\n')
        for i in range(len(lines)//2):
            games.append(lines[2*i] + '\n\n' + lines[2*i+1] + '\n\n')

        print("finished formatting games for file", pgn.name)
            
        counter = 0 

        for cur_game in games:
            counter += 1
            if counter % 1000 == 0:
                print("processing game", counter)
                print("data_counter", data_counter)

            # error_buffer = io.StringIO()
            # sys.stderr = error_buffer
            game = chess.pgn.read_game(io.StringIO(cur_game))
            # sys.stderr = sys.__stderr__
            # error_message = error_buffer.getvalue().strip()
            # if error_message:
            #     print(cur_game)
            #     exit()

            # game.headers have all sorts of information.
            if game.headers['Result'] == '1-0' or game.headers['Result'] == '0-1' or game.headers['Result'] == '1/2-1/2':
                #  check if white player is rated
                # check that mainline_moves() is not empty

                # select N random moves from the game
                N = 3

                if 'WhiteElo' in game.headers and 'BlackElo' in game.headers and int(game.headers['WhiteElo']) > 1000 and int(game.headers['BlackElo']) > 1000 and len(list(game.mainline_moves())) > 2:
                    board = game.board()
                    i = 0
                    # sample moves
                    mv = random.sample(list(game.mainline_moves()), N)
                    for i in range(N):
                        mv[i] = mv[i].uci()

                    for move in game.mainline_moves():
                        if not board.is_legal(move):
                            print(game.headers['Site'], move, board.is_legal(move), board.is_game_over())
                            exit()
                        
                        board.push(move)
                        if move.uci() in mv:
                            boards.append(Board(board, game.headers['Result']))
                            data_counter += 1

            while(len(boards) > GAME_PER_FILE):
                # pop off the first 100 boards
                temp_boards = [board.board_map for board in boards[:GAME_PER_FILE]]
                boards = boards[GAME_PER_FILE:]

                # save the first 100 boards to a file
                np.save(PATH + str(file_counter*GAME_PER_FILE) + '.npy', temp_boards)
                print("saved", file_counter*GAME_PER_FILE, "games")
                file_counter += 1
            

            if file_counter * GAME_PER_FILE > 3000000:
                return boards, file_counter * GAME_PER_FILE
            
    return boards, file_counter * GAME_PER_FILE

if __name__ == '__main__':
    boards, dat = get_boards(get_pgn_files(PATH))
    print( dat)
    print(boards[0].board_map[0,:,:])
    
