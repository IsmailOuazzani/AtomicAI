import chess
import chess.pgn

import random
import time

# TODO: add multithreading for search
# TODO: replace eval with NN eval using ONNX
# TODO: save MCTS tree to do less evaluation
# TODO: baseline engine blunders hard :c
# TODO: replace random move selection in simulation with max eval move


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
        return self.wins / self.visits + 2 * (2 * self.parent.visits / self.visits) ** 0.5

class MCTSEngine:
    def __init__(self, time_limit=30.0):
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
                node.wins += result + 0.5 * evaluate(temp_board, 0) # Add evaluation score
                node = node.parent
        best_node = max(root.children, key=lambda child: child.visits)
        return best_node.move

    def simulate(self, board):
        result = None
        while not board.is_game_over(claim_draw=True):
            legal_moves = list(board.legal_moves)
            # move = random.choice(legal_moves)

            # choose move with max eval
            # sample 50 random moves to reduce computation
            legal_test_moves = random.sample(legal_moves, min(50, len(legal_moves)))
            states = [board.copy() for _ in range(len(legal_test_moves))]
            for i, state in enumerate(states):
                state.push(legal_test_moves[i])
            evals = [evaluate(state, 0) for state in states]
            move = legal_test_moves[evals.index(max(evals))]

            board.push(move)
            if board.is_checkmate():
                result = 1 if not board.turn else -1
                break
            if board.is_variant_win():
                result = 1 if not board.turn else -1
                break
            if board.is_variant_loss():
                result = -1 if not board.turn else 1
                break
            if board.is_stalemate():
                result = 0
                break
            if board.is_variant_draw():
                result = 0
                break
        if result is None:
            result = 0
        return result

def evaluate(board, player):
    # material score
    score = 0
    for piece in board.piece_map().values():
        if piece.color == player:
            score += piece.piece_type
        else:
            score -= piece.piece_type

    # missing king score
    if not board.king(player):
        score -= 1000
    if not board.king(not player):
        score += 1000
    # normalise score
    score /= 39
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
        move = engine.choose_move(board)
        print(f"Time elapsed: {time.time() - start_time:.2f}s")
        board.push(move)
    # save as svg
    board_svg = board._repr_svg_()
    with open("board.svg", "w+") as f:
        f.write(board_svg)
    print("player eval:", evaluate(board, 1))

print(board)
print("Game over", board.result())