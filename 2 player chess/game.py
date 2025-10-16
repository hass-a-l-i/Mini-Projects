import game_state
from game_state import *


def game():
    global piece_pos
    running = True
    board = game_state.BoardState()
    print("Welcome to Chess. Type \"QUIT\" at any time to exit.")
    print(board)
    while running:
        while True:
            check = False
            if len(board.move_log) > 0:
                prev_move_log = board.move_log[-1]
                prev_move = prev_move_log[2]
                prev_piece = prev_move_log[0]
                prev_move_pos_y, prev_move_pos_x = notation_to_index(prev_move)
                check = current_check(board, prev_move_pos_y, prev_move_pos_x, board.move_log)
                if check is True and prev_piece != "..":
                    check_mate = checkmate(prev_piece, board, board.move_log)
                    if check_mate is True:
                        if prev_piece[0] == "w":
                            print("Checkmate! White wins!")
                        else:
                            print("Checkmate! Black wins!")
                        running = False
                        break
                    if prev_piece[0] == "w":
                        print("Black king check. Protect your king!")
                        board.white_turn = False
                    else:
                        print("White king check. Protect your king!")
                        board.white_turn = True
                else:
                    stale_mate = stalemate(board, board.move_log)
                    if stale_mate is True:
                        print("Stalemate!")
                        running = False
                        break
            if running is False:
                break
            print("Choose piece position:")
            piece_pos = input()
            if piece_pos == "QUIT":
                running = False
                break
            piece_pos_y, piece_pos_x = notation_to_index(piece_pos)
            if piece_pos_y == -1 and piece_pos_x  == -1:
                print("Square out of bounds. Try again.")
                break
            piece = board[piece_pos_y, piece_pos_x]
            piece_obj = Piece.create(piece[0], piece[1], piece_pos)
            if (piece_pos_y > 7 or piece_pos_x > 7) or (piece_pos_y < 0 or piece_pos_x < 0):
                print("Square out of bounds. Try again.")
                break
            elif piece == "..":
                print(f"{piece_pos} is empty. Try again.")
                break
            elif board.white_turn is True and piece_obj.color == "b":
                print("It is currently whites turn to move. Please select a white piece.")
                break
            elif board.white_turn is False and piece_obj.color == "w":
                print("It is currently blacks turn to move. Please select a black piece.")
                break
            else:
                print(f"{piece} at {piece_pos} selected.")
            if running is False:
                break

            print("Choose move position:")
            move_pos = input()
            if move_pos == "QUIT":
                running = False
                break
            move_pos_y, move_pos_x = notation_to_index(move_pos)
            if (move_pos_y > 7 or move_pos_x > 7) or (move_pos_y < 0 or move_pos_x < 0):
                print("Square out of bounds. Try again.")
                break
            captured = board[move_pos_y][move_pos_x]
            check_future = False
            if check is True and len(board.move_log) > 0:
                check_future = future_check(board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, board.move_log)
            check_self = self_check(board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, board.move_log)
            check_2 = check_future or check_self
            piece_pos_y, piece_pos_x = notation_to_index(piece_pos)
            piece = board[piece_pos_y, piece_pos_x]
            move_valid = board.make_move(piece_pos, move_pos)
            if move_valid is True and check_2 is False:
                board.move_log.append((piece, piece_pos, move_pos, captured))
                print(board)
                print(board.move_log)
                check = False
                break
            else:
                print(f"{piece_pos} to {move_pos} not valid. Try again.")
        if running is False:
            break
    print("Game Over")



