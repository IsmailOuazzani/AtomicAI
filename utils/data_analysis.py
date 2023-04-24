# Take in a pgn file
# Output the binary map of each board after each move in the game

# TODO: test castling and en passant in board representation
# TODO: decaying result label + elo trust

import chess
import chess.pgn
import numpy as np
import os
import io
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

GAME_PER_FILE = 5000 # 10 000 is too slow

# games are located in the dataset folder in the parent directory in the form of pgn files
# games need to be uncompressed first (see README.md)
PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/')

# read pgns
def get_pgn_files(path):
    pgn_files = []
    for file in os.listdir(path):
        if file.endswith('.pgn'):
            pgn_files.append(open(path + file, 'r'))
    print("Length of pgn_files = ", len(pgn_files))
    #print(pgn_files)
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
    white_wins, black_wins, draws, invalids, abandoned = 0, 0, 0, 0, 0

    for pgn in pgn_files[3:4]:
        # handle multiple games in one pgn file
        file_counter+=1
        
        games = []
        pgn_read = pgn.read()
        # replace \n\n\n with \n\n
        pgn_clean = pgn_read.replace('\n\n\n\n', '\n\n')
        pgn_clean = pgn_clean.replace('\n\n\n', '\n\n')
        lines = pgn_clean.split('\n\n')

        for i in range(300000):
            
            games.append(lines[2*i] + '\n\n' + lines[2*i+1] + '\n\n')
            if i % GAME_PER_FILE == 0:
                print("formatting game", i)

        # print("finished formatting games for file", pgn.name)
    
          
        counter = 0 
        EloTotal = []
        Duration = []
        Openers = {}
        
        print("Total number of games: ",len(games))
        for cur_game in games:
            counter += 1
            if counter % 100000 == 0:
                print("processing game", counter)

            game = chess.pgn.read_game(io.StringIO(cur_game))
            
            #print("Game Moves for game 1")
            Duration.append(len(list(game.mainline_moves()))//2)

            #print(counter)
            
            moves = list(game.mainline_moves())
            if moves:
                opener = moves[0]
                if str(opener) not in Openers.keys():
                    Openers[str(opener)] = 1
                else:
                    Openers[str(opener)] += 1
            else:
                abandoned +=1 
            
            #print(opener)
            
           
           
           
            
            if game.headers['Result'] == '1-0':
                #print("whitey")
                white_wins+=1
            elif game.headers['Result'] == '0-1':
                black_wins+=1
            elif game.headers['Result'] == '1/2-1/2':
                draws+=1
            else:
                invalids+=1
            
            EloTotal.append(int(game.headers['WhiteElo']))
            EloTotal.append(int(game.headers['BlackElo']))
        
        #print("length of elo array:", len(EloTotal))   
        #print(statistics.mean(Duration))

        elo_dist = sns.histplot(x=EloTotal, stat="probability", bins = 20)
        elo_dist.set_title('Elo Distribution')
        elo_dist.set_xlabel('Elo Rating')
        elo_dist.set_ylabel('Frequency')
        plt.show()

        dur_dist = sns.histplot(x=Duration, stat="probability", bins = 20)
        dur_dist.set_title('Duration Distribution')
        dur_dist.set_xlabel('Duration')
        dur_dist.set_ylabel('Frequency')
        plt.show()

        sorted_openers = sorted(Openers.items(), key=lambda x:x[1], reverse=True)
        sorted_op_dict = dict(sorted_openers)
        print("Here it comes... most common openings")
        labels = list(sorted_op_dict.keys())[0:5]
        data = list(sorted_op_dict.values())[0:5]
        colors = sns.color_palette('pastel')[0:5]
        plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
        plt.show()
        
            
    
    return counter, white_wins, black_wins, draws, invalids, abandoned, file_counter

if __name__ == '__main__':
    
    print(get_boards(get_pgn_files(PATH)))
