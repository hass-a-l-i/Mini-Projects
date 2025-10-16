from game import *

if __name__ == "__main__":
    game()

    """
    board = game_state.BoardState()
    board_2 = board.board.flatten()
    print(board_2)
    board_set = set(board_2)
    set2 = {'bK', '..', 'wK'}
    set3 = {'bK', '..', 'wK', "wN"}
    sets = [set2, set3]
    print(board_set)
    if "wK" in board_set:
        print("Y")
    if board_set == set2:
        print("Y2")
    if board_set in sets:
        print("Y3")


    board = game_state.BoardState()
    board_2 = board.board
    board_highlight = highlight_moves("E8", board_2, board.move_log)
    print(board_highlight.flatten())
    print(set(board_highlight.flatten()))"""

