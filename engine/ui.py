# inspired by the https://github.com/thomasahle/sunfish user inferface

import chess
import chess.pgn
import chess.svg
import argparse
from chessnet2 import next_move_chessnet2, evaluate_board

def render_board(board, move, user_side):
    if move is not None:
        board_svg = chess.svg.board(board=board, size=400, orientation=user_side, lastmove=move)
    else:
        board_svg = chess.svg.board(board=board, size=400, orientation=user_side)
    with open("board.svg", "w+") as f:
        f.write(board_svg)

def start():
    # intialize board to atomic varient
    board = chess.pgn.read_game(open("starter.pgn")).board()
    print(board.uci_variant)

    # print evaluation of starting position
    print("Starting position evaluation:", evaluate_board(board))
    
    # if play against a human
    human = True
    if not human:
        from movegeneration import next_move as get_move
    else:
        def get_move(depth, board: chess.Board) -> chess.Move:
            move = input(f"\nYour move (e.g. {list(board.legal_moves)[0]}):\n")

            for legal_move in board.legal_moves:
                if move == str(legal_move):
                    return legal_move
            return get_move(board)

    user_side = (
        chess.WHITE if input("Start as [w]hite or [b]lack:\n") == "w" else chess.BLACK
    )


    render_board(board, None, user_side)


    if user_side == chess.WHITE:
        print(render(board))
        print()
        if human:
            move = get_move(get_depth(), board)
            player = 'Human'
        else:
            # move generated by baseline
            move = get_move(get_depth(),board)
            player = 'Baseline'
        print(player, "as white, vs AI as black")

        board.push(move)
        render_board(board, move, user_side)
    else:
        print("AI as white, vs","human" if human else "Baseline"," as black")

    while not board.is_game_over():
        print('My Turn HAHHAHA')
        move, eval = next_move_chessnet2(get_depth(), board, debug=False)
        board.push(move)
        render_board(board, move, user_side)
        print(render(board))
        print(f"info {move} White eval {eval}") #AI eval
        # opposite player's turn
        if not board.is_game_over():
            move = get_move(get_depth(),board)
            # move, eval = next_move_chessnet2(get_depth(), board, debug=False)
            board.push(move)
            render_board(board, move, user_side)

        

    print(f"\nResult: [w] {board.result()} [b]")
    if board.result() == "1-0" and user_side == chess.WHITE:
        print(player, "wins!")
    elif board.result() == "0-1" and user_side == chess.BLACK:
        print(player, "wins!")
    else:
        print("AI wins!")

    # save game as pgn
    pgn = open("game.pgn", "w+")
    exporter = chess.pgn.FileExporter(pgn)
    game = chess.pgn.Game()
    game.headers["Event"] = "Chessnet2"
    game.headers["Site"] = ""
    game.headers["Date"] = ""
    game.headers["Round"] = ""
    game.headers["White"] = "Human" if human else "Baseline"
    game.headers["Black"] = "AI"
    game.headers["Result"] = board.result()
    game.add_line(board.move_stack)
    game.accept(exporter)
    pgn.close()

    


def render(board: chess.Board) -> str:
    """
    Print a side-relative chess board with special chess characters.
    """
    board_string = list(str(board))
    uni_pieces = {
        "R": "R",
        "N": "N",
        "B": "B",
        "Q": "Q",
        "K": "K",
        "P": "P",
        "r": "r",
        "n": "n",
        "b": "b",
        "q": "q",
        "k": "k",
        "p": "p",
        ".": "·",
    }
    for idx, char in enumerate(board_string):
        if char in uni_pieces:
            board_string[idx] = uni_pieces[char]
    ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]
    display = []
    for rank in "".join(board_string).split("\n"):
        display.append(f"  {ranks.pop()} {rank}")
    if board.turn == chess.BLACK:
        display.reverse()
    display.append("    a b c d e f g h")
    return "\n" + "\n".join(display)



def get_depth() -> int:
    return 3


if __name__ == "__main__":
    start()
