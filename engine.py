import chess
import random
import time

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
    def __init__(self, time_limit=1.0):
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
                node.wins += result + 0.1 * self.evaluate(temp_board) # Add evaluation score
                node = node.parent
        best_node = max(root.children, key=lambda child: child.visits)
        return best_node.move

    def simulate(self, board):
        result = None
        while not board.is_game_over(claim_draw=True):
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            board.push(move)
            if board.is_checkmate():
                result = 1 if board.turn else -1
                break
        if result is None:
            result = 0
        return result

    def evaluate(self, board):
        # material score
        score = 0
        for piece in board.piece_map().values():
            if piece.color:
                score += piece.piece_type
            else:
                score -= piece.piece_type
        return score / 2*8*2


engine = MCTSEngine()

board = chess.Board()
while not board.is_game_over(claim_draw=True):
    if board.turn:
        print(board)
        print("Your turn")
        move = input("Enter your move (e.g. e2e4): ")
        board.push_san(move)
    else:
        print(board)
        print("Engine thinking...")
        start_time = time.time()
        move = engine.choose_move(board)
        print(f"Time elapsed: {time.time() - start_time:.2f}s")
        board.push(move)

print(board)
print("Game over")