import chess
import chess.pgn

import random
import time

# TODO: add multithreading for search
# TODO: replace eval with NN eval using ONNX
# TODO: save MCTS tree to do less evaluation
# TODO: baseline engine blunders hard :c
# TODO: replace random move selection in simulation with max eval move


# TODO: not sure if this is the best way to do this, but can try:
# 1. kernel functions in input of CNN


class Node:
    def __init__(self, parent, move):
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0
        self.children = []

    def ucb1(self):
        if self.visits == 0:
            return float("inf")
        exploit = self.wins / self.visits
        explore_constant = 2.0
        return exploit + explore_constant * (2 * self.parent.visits / self.visits) ** 0.5

class MCTSEngine:
    def __init__(self, time_limit=60.0):
        self.time_limit = time_limit

    def choose_move(self, board):
        root = Node(None, None)
        start_time = time.monotonic()
        while time.monotonic() - start_time < self.time_limit:
            node = root
            temp_board = board.copy()
            while node.children:
                node = max(node.children, key=lambda child: child.ucb1())
                temp_board.push(node.move)
            if node.visits == 0:
                legal_moves = list(temp_board.legal_moves)
                move = random.choice(legal_moves)
                node.visits += 1
                child = Node(node, move)
                node.children.append(child)
                node = child
                temp_board.push(child.move)
                result = self.simulate(temp_board)
            else:
                if node.children:
                    child = max(node.children, key=lambda child: child.ucb1())
                    temp_board.push(child.move)
                    result = self.simulate(temp_board)
                else:
                    result = self.simulate(temp_board)
            while node is not None:
                node.visits += 1
                node.wins += result
                node = node.parent
        best_node = max(root.children, key=lambda child: child.visits)
        return best_node.move, best_node.wins

    def simulate(self, test_board, testall=False):
        result = None
        board = test_board.copy()
        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            # move = random.choice(legal_moves)

            # choose move with max eval
            # sample 500 random moves to reduce computation
            legal_test_moves = random.sample(legal_moves, min(100, len(legal_moves)))
            if testall:
                legal_test_moves = legal_moves
            states = [board.copy() for _ in range(len(legal_test_moves))]
            for i, state in enumerate(states):
                state.push(legal_test_moves[i])
            evals = [evaluate(state, not state.turn) for state in states]
            move = legal_test_moves[evals.index(max(evals))]

            board.push(move)
        # debugging simulation
        if testall:
            # save final board as svg
            board_svg = board._repr_svg_()
            with open("board_final.svg", "w+") as f:
                f.write(board_svg)
            # print("game", board.is_game_over(), board.result())

        # Determine the final result
        if board.result() == "0-1":
            return 1
        elif board.result() == "1-0":
            return -1
        else:
            return 0

def evaluate(board, player):
    # material score
    score = 0
    for piece in board.piece_map().values():
        if piece.color == player:
            score += piece.piece_type
        else:
            score -= piece.piece_type

    if player == 1:
        enemy = chess.BLACK
        friend = chess.WHITE
    else:
        enemy = chess.WHITE
        friend = chess.BLACK

    # mobility score
    score += len(list(board.legal_moves)) / 10

    # king safety score
    if board.king(enemy) is not None:
        score += len(board.attackers(enemy, board.king(enemy))) /2

    # friendly pieces closer to enemy king
    if board.king(enemy) is not None and board.king(player) is not None:
        for i in range(8):
            for j in range(8):
                if board.piece_at(chess.square(i, j)) is not None:
                    if board.piece_at(chess.square(i, j)).color == friend:
                        score += 1 / (abs(i - chess.square_file(board.king(enemy))) + abs(j - chess.square_rank(board.king(enemy))))
                    elif board.piece_at(chess.square(i, j)).color == enemy:
                        score -= 1 / (abs(i - chess.square_file(board.king(friend))) + abs(j - chess.square_rank(board.king(friend))))
    

    # exploding king score
    if board.king(friend) is None:
        score -= 100
    if board.king(enemy) is None:
        score += 100

    # normalise score
    score /= 40
    return score 

engine = MCTSEngine()
# create board from starter.pgn since i can't figure out how to create a variant board
board = chess.pgn.read_game(open("starter.pgn")).board()
while not board.is_game_over(claim_draw=True):
    if board.turn:
        print(board)
        print("Your turn")
        move = input("Enter your move (e.g. e2e4): ")
        # check if move is legal
        while move not in [move.uci() for move in board.legal_moves]:
            print("Illegal move")
            move = input("Enter your move (e.g. e2e4): ")
        board.push_san(move)
    else:
        print(board)
        print("Engine thinking...")
        start_time = time.time()
        move, boardeval = engine.choose_move(board)
        print(f"Time elapsed: {time.time() - start_time:.2f}s")
        # print("Engine eval:", boardeval)
        board.push(move)
    # save as svg
    board_svg = board._repr_svg_()
    with open("board.svg", "w+") as f:
        f.write(board_svg)
    print("engine eval:", evaluate(board, 0))
    print("engine simulate:", engine.simulate(board.copy(), testall=True))

print(board)
print("Game over", board.result())