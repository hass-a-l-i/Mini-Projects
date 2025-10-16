import numpy as np
from helpers import *


class BoardState:
    def __init__(self):
        self.board = np.array((
                             ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
                             ["..", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
                             ["..", "..", "..", "..", "..", "..", "..", ".."],
                             ["..", "..", "..", "..", "..", "..", "..", ".."],
                             ["..", "..", "..", "..", "..", "..", "..", ".."],
                             ["..", "..", "..", "..", "..", "..", "..", ".."],
                             ["..", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
                             ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
                            ))
        self.move_log = []
        self.white_turn = True

    def __getitem__(self, key):
        return self.board[key]

    def __str__(self):
        return str(self.board)

    def flatten(self):
        return self.board.flatten().tolist()

    def copy(self):
        return self.board.copy()

    def make_move(self, piece_pos, move_pos):
        piece_pos_y, piece_pos_x = notation_to_index(piece_pos)
        move_pos_y, move_pos_x = notation_to_index(move_pos)
        piece = self.board[piece_pos_y, piece_pos_x]
        move = self.board[move_pos_y, move_pos_x]
        piece_obj = Piece.create(piece[0], piece[1], piece_pos)
        move_valid = piece_obj.valid_move(self.board, piece_pos_y, piece_pos_x,
                                          move_pos_y, move_pos_x, piece, move, self.move_log)
        if piece_obj.type == "K":
            if move_valid is False:
                move_valid, king_side = piece_obj.castle(self.board, piece_pos_y, piece_pos_x,
                                                         move_pos_y, move_pos_x, piece, self.move_log)
                if move_valid is True:
                    if king_side is True:
                        self.board[move_pos_y][move_pos_x - 1] = self.board[move_pos_y][move_pos_x + 1]
                        self.board[move_pos_y][move_pos_x + 1] = ".."
                    else:
                        self.board[move_pos_y][move_pos_x + 1] = self.board[move_pos_y][move_pos_x - 2]
                        self.board[move_pos_y][move_pos_x - 2] = ".."
        check_2 = False
        check_future = False
        if len(self.move_log) > 0:
            prev_move_log = self.move_log[-1]
            prev_move = prev_move_log[2]
            prev_move_pos_y, prev_move_pos_x = notation_to_index(prev_move)
            check = current_check(self.board, prev_move_pos_y, prev_move_pos_x, self.move_log)
            if check is True:
                check_future = future_check(self.board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, self.move_log)
            check_self = self_check(self.board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, self.move_log)
            check_2 = check_future or check_self
        if move_valid is True and check_2 is False:
            self.board[move_pos_y][move_pos_x] = self.board[piece_pos_y][piece_pos_x]
            self.board[piece_pos_y][piece_pos_x] = ".."
            self.white_turn = not self.white_turn
        return move_valid


class Piece:
    def __init__(self, color, type, piece_pos):
        self.color = color
        self.type = type
        self.piece_pos = piece_pos

    def color_check(self, move):
        move_valid = True
        if move[0] == self.color:
            move_valid = False
        return move_valid

    def directions(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x,
                   move, check_list, sign_x, sign_y):
        takes = False
        move_valid = False
        for k in range(0, 7):
            new_x_pos = move_pos_x + (sign_x * k)
            new_y_pos = move_pos_y + (sign_y * k)
            if new_x_pos == piece_pos_x and new_y_pos == piece_pos_y:
                break
            check_square = board[new_y_pos][new_x_pos]
            check_list.append(check_square)
        if len(set(check_list)) == 1:
            if check_list[0] == "..":
                move_valid = True
            elif len(check_list) == 1 and self.color_check(move) is True:
                move_valid = True
                takes = True
        elif len(set(check_list)) == 2 and check_list.count(move) == 1 and self.color_check(move) is True:
            if check_list[-1] == "..":
                move_valid = True
                takes = True
        return move_valid, takes

    @classmethod
    def create(cls, color, type, piece_pos):
        if type == "P":
            return Pawn(color, type, piece_pos)
        elif type == "K":
            return King(color, type, piece_pos)
        elif type == "R":
            return Rook(color, type, piece_pos)
        elif type == "B":
            return Bishop(color, type, piece_pos)
        elif type == "Q":
            return Queen(color, type, piece_pos)
        elif type == "N":
            return Knight(color, type, piece_pos)


class Pawn(Piece):
    def en_passant_check(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, move, move_log):
        move_valid = False
        del_pos_x, del_pos_y = -1, -1
        pos_change_y = move_pos_y - piece_pos_y
        pos_change_x = move_pos_x - piece_pos_x

        left_piece = board[piece_pos_y][piece_pos_x - 1]
        right_piece = board[piece_pos_y][piece_pos_x + 1]

        prev_piece = move_log[-1][0]
        prev_piece_pos = move_log[-1][1]
        prev_move_pos = move_log[-1][2]

        prev_move_pos_y, prev_move_pos_x = notation_to_index(prev_move_pos)
        prev_piece_pos_y, prev_piece_pos_x = notation_to_index(prev_piece_pos)
        prev_pos_change_y = prev_move_pos_y - prev_piece_pos_y
        taken_pos_change = piece_pos_x - prev_move_pos_x

        if move == "..":
            if self.color == "w" and prev_piece == "bP" and prev_pos_change_y == 2:
                if pos_change_y == -1 and pos_change_x == -1 and left_piece == "bP" and taken_pos_change == 1:
                    move_valid = True
                    del_pos_x = piece_pos_x - 1
                    del_pos_y = piece_pos_y
                elif pos_change_y == -1 and pos_change_x == 1 and right_piece == "bP" and taken_pos_change == -1:
                    move_valid = True
                    del_pos_x = piece_pos_x + 1
                    del_pos_y = piece_pos_y
            elif self.color == "b" and prev_piece == "wP" and prev_pos_change_y == -2:
                if pos_change_y == 1 and pos_change_x == -1 and left_piece == "wP" and taken_pos_change == 1:
                    move_valid = True
                    del_pos_x = piece_pos_x - 1
                    del_pos_y = piece_pos_y
                elif pos_change_y == 1 and pos_change_x == 1 and right_piece == "wP" and taken_pos_change == -1:
                    move_valid = True
                    del_pos_x = piece_pos_x + 1
                    del_pos_y = piece_pos_y
        return move_valid, del_pos_x, del_pos_y

    def takes(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, move, move_log):
        take = False
        opp_color = self.color_check(move)
        pos_change_y = move_pos_y - piece_pos_y
        pos_change_x = move_pos_x - piece_pos_x
        if move != ".." and opp_color is True and (pos_change_x == 1 or pos_change_x == -1):
            if self.color == "w":
                if pos_change_y == -1:
                    take = True
            elif self.color == "b":
                if pos_change_y == 1:
                    take = True
        return take

    def valid_move(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, piece, move, move_log):
        move_valid = False
        promotion = False
        en_passant = False
        opp_color = self.color_check(move)
        pos_change_y = move_pos_y - piece_pos_y
        pos_change_x = move_pos_x - piece_pos_x
        if move_pos_x == piece_pos_x:
            if self.color == "w":
                if move == "..":
                    if pos_change_y == -1 and move_pos_y > 0:
                        move_valid = True
                    elif pos_change_y == -1 and move_pos_y == 0:
                        promotion = True
                    elif pos_change_y == -2 and piece_pos_y == 6 and board[piece_pos_y - 1][piece_pos_x] == "..":
                        move_valid = True
            elif self.color == "b":
                if move == "..":
                    if pos_change_y == 1 and move_pos_y < 7:
                        move_valid = True
                    elif pos_change_y == 1 and move_pos_y == 7:
                        promotion = True
                    elif pos_change_y == 2 and piece_pos_y == 1 and board[piece_pos_y + 1][piece_pos_x] == "..":
                        move_valid = True
        elif move != ".." and opp_color is True and (pos_change_x == 1 or pos_change_x == -1):
            if self.color == "w":
                if pos_change_y == -1:
                    move_valid = True
                    if move_pos_y == 0:
                        promotion = True
            elif self.color == "b":
                if pos_change_y == 1:
                    move_valid = True
                    if move_pos_y == 7:
                        promotion = True
        if promotion is True:
            promotion_list = ["B", "N", "Q", "R"]
            print("Please select your piece for promotion. Type one letter : B, N, R, Q")
            while True:
                promote_choice = input()
                if promote_choice in promotion_list:
                    board[piece_pos_y][piece_pos_x] = self.color + promote_choice
                    break
                else:
                    print("Invalid selection, please try again.")
        if len(move_log) > 0:
            en_passant, del_pos_x, del_pos_y = self.en_passant_check(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                                     move_pos_x, move, move_log)
            if en_passant is True:
                board[del_pos_y][del_pos_x] = ".."
        move_valid = move_valid or promotion or en_passant
        return move_valid


class King(Piece):
    def castle(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, piece, move_log):
        move_valid = False
        king_side = False
        pos_change_x = move_pos_x - piece_pos_x
        pieces, move_from, move_to, captured = zip(*move_log)
        rook_check = zip(pieces, move_from)
        if self.color == "w" and piece_pos_y == move_pos_y:
            if pos_change_x == 2 and board[piece_pos_y][piece_pos_x + 1] == "..":
                check = self.check_move_king(board, piece_pos_y, piece_pos_x, piece_pos_y, piece_pos_x + 1, move_log)
                if board[piece_pos_y][piece_pos_x + 2] == ".." and piece not in pieces and check is False:
                    board_copy = board.copy()
                    board_copy[piece_pos_y][piece_pos_x + 1] = board_copy[piece_pos_y][piece_pos_x]
                    board_copy[piece_pos_y][piece_pos_x] = ".."
                    check = self.check_move_king(board_copy, piece_pos_y, piece_pos_x + 1, piece_pos_y, piece_pos_x + 2,
                                                 move_log)
                    if board[piece_pos_y][piece_pos_x + 3] == "wR" and ("wR", "H1") not in rook_check and check is False:
                        move_valid = True
                        king_side = True
            elif pos_change_x == -2 and board[piece_pos_y][piece_pos_x - 1] == "..":
                check = self.check_move_king(board, piece_pos_y, piece_pos_x, piece_pos_y, piece_pos_x - 1, move_log)
                if board[piece_pos_y][piece_pos_x - 2] == ".." and piece not in pieces and check is False:
                    board_copy = board.copy()
                    board_copy[piece_pos_y][piece_pos_x - 1] = board_copy[piece_pos_y][piece_pos_x]
                    board_copy[piece_pos_y][piece_pos_x] = ".."
                    check = self.check_move_king(board_copy, piece_pos_y, piece_pos_x - 1, piece_pos_y, piece_pos_x - 2,
                                                 move_log)
                    if board[piece_pos_y][piece_pos_x - 3] == ".." and ("wR", "A1") not in rook_check and check is False:
                        if board[piece_pos_y][piece_pos_x - 4] == "wR":
                            move_valid = True
                            king_side = False
        elif self.color == "b" and piece_pos_y == move_pos_y:
            if pos_change_x == 2 and board[piece_pos_y][piece_pos_x + 1] == "..":
                check = self.check_move_king(board, piece_pos_y, piece_pos_x, piece_pos_y, piece_pos_x + 1, move_log)
                if board[piece_pos_y][piece_pos_x + 2] == ".." and piece not in pieces and check is False:
                    board_copy = board.copy()
                    board_copy[piece_pos_y][piece_pos_x + 1] = board_copy[piece_pos_y][piece_pos_x]
                    board_copy[piece_pos_y][piece_pos_x] = ".."
                    check = self.check_move_king(board_copy, piece_pos_y, piece_pos_x + 1, piece_pos_y, piece_pos_x + 2,
                                                 move_log)
                    if board[piece_pos_y][piece_pos_x + 3] == "bR" and ("bR", "H8") not in rook_check and check is False:
                        move_valid = True
                        king_side = True
            elif pos_change_x == -2 and board[piece_pos_y][piece_pos_x - 1] == "..":
                check = self.check_move_king(board, piece_pos_y, piece_pos_x, piece_pos_y, piece_pos_x - 1, move_log)
                if board[piece_pos_y][piece_pos_x - 2] == ".." and piece not in pieces and check is False:
                    board_copy = board.copy()
                    board_copy[piece_pos_y][piece_pos_x - 1] = board_copy[piece_pos_y][piece_pos_x]
                    board_copy[piece_pos_y][piece_pos_x] = ".."
                    check = self.check_move_king(board_copy, piece_pos_y, piece_pos_x - 1, piece_pos_y, piece_pos_x - 2,
                                                 move_log)
                    if board[piece_pos_y][piece_pos_x - 3] == ".." and ("bR", "A8") not in rook_check and check is False:
                        if board[piece_pos_y][piece_pos_x - 4] == "bR":
                            move_valid = True
                            king_side = False
        return move_valid, king_side

    def check_move_king(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, move_log):
        king_piece = board[piece_pos_y][piece_pos_x]
        for i, row in enumerate(board):
            for j, square in enumerate(row):
                piece = board[i, j]
                if piece != ".." and self.color != piece[0]:
                    board_copy = board.copy()
                    board_copy[move_pos_y][move_pos_x] = board_copy[piece_pos_y][piece_pos_x]
                    board_copy[piece_pos_y][piece_pos_x] = ".."
                    piece_notation = index_to_notation(i, j)
                    piece_obj = Piece.create(piece[0], piece[1], piece_notation)
                    if piece_obj.takes(board_copy, i, j, move_pos_y, move_pos_x, king_piece, move_log):
                        return True
        return False

    def takes(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, move, move_log):
        take = False
        opp_color = self.color_check(move)
        pos_change_y = move_pos_y - piece_pos_y
        pos_change_x = move_pos_x - piece_pos_x
        up_down = abs(pos_change_x) + abs(pos_change_y)
        diag = abs(pos_change_x) * abs(pos_change_y)
        if (up_down == 1 or diag == 1) and opp_color is True:
            take = True
        return take

    def valid_move(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, piece, move, move_log):
        move_valid = False
        opp_color = self.color_check(move)
        pos_change_y = move_pos_y - piece_pos_y
        pos_change_x = move_pos_x - piece_pos_x
        up_down = abs(pos_change_x) + abs(pos_change_y)
        diag = abs(pos_change_x) * abs(pos_change_y)
        if move == "..":
            if up_down == 1 or diag == 1:
                move_valid = True
        elif (up_down == 1 or diag == 1) and opp_color is True:
            move_valid = True
        return move_valid


class Rook(Piece):
    def takes(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, move, move_log):
        takes = False
        direction_up = False
        direction_right = False
        if move_pos_y - piece_pos_y < 0:
            direction_up = True
        if move_pos_x - piece_pos_x > 0:
            direction_right = True
        check_list = []
        if self.type == "R" or self.type == "Q":
            if move_pos_y == piece_pos_y:
                if direction_right is True:
                    move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                        move_pos_x, move, check_list, -1, 0)
                else:
                    move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                        move_pos_x, move, check_list, 1, 0)
            elif move_pos_x == piece_pos_x:
                if direction_up is True:
                    move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                        move_pos_x, move, check_list, 0, 1)
                else:
                    move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                        move_pos_x, move, check_list, 0, -1)
        return takes

    def valid_move(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, piece, move, move_log):
        move_valid = False
        direction_up = False
        direction_right = False
        if move_pos_y - piece_pos_y < 0:
            direction_up = True
        if move_pos_x - piece_pos_x > 0:
            direction_right = True
        check_list = []
        if self.type == "R" or self.type == "Q":
            if move_pos_y == piece_pos_y:
                if direction_right is True:
                    move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                 move_pos_x, move, check_list, -1, 0)
                else:
                    move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                 move_pos_x, move, check_list, 1, 0)
            elif move_pos_x == piece_pos_x:
                if direction_up is True:
                    move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                 move_pos_x, move, check_list, 0, 1)
                else:
                    move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                 move_pos_x, move, check_list, 0, -1)
        return move_valid


class Bishop(Piece):
    def takes(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, move, move_log):
        takes = False
        ab_pos_change_y = abs(move_pos_y - piece_pos_y)
        ab_pos_change_x = abs(move_pos_x - piece_pos_x)
        direction_up = False
        direction_right = False
        if move_pos_y - piece_pos_y < 0:
            direction_up = True
        if move_pos_x - piece_pos_x > 0:
            direction_right = True
        check_list = []
        if (self.type == "B" or self.type == "Q") and ab_pos_change_x == ab_pos_change_y:
            if direction_right is True and direction_up is True:
                move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                    move_pos_x, move, check_list, -1, 1)
            elif direction_right is False and direction_up is True:
                move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                    move_pos_x, move, check_list, 1, 1)
            elif direction_right is True and direction_up is False:
                move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                    move_pos_x, move, check_list, -1, -1)
            elif direction_right is False and direction_up is False:
                move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                                    move_pos_x, move, check_list, 1, -1)
        return takes

    def valid_move(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, piece, move, move_log):
        move_valid = False
        ab_pos_change_y = abs(move_pos_y - piece_pos_y)
        ab_pos_change_x = abs(move_pos_x - piece_pos_x)
        direction_up = False
        direction_right = False
        if move_pos_y - piece_pos_y < 0:
            direction_up = True
        if move_pos_x - piece_pos_x > 0:
            direction_right = True
        check_list = []
        if (self.type == "B" or self.type == "Q") and ab_pos_change_x == ab_pos_change_y:
            if direction_right is True and direction_up is True:
                move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                             move_pos_x, move, check_list, -1, 1)
            elif direction_right is False and direction_up is True:
                move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                             move_pos_x, move, check_list, 1, 1)
            elif direction_right is True and direction_up is False:
                move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                             move_pos_x, move, check_list, -1, -1)
            elif direction_right is False and direction_up is False:
                move_valid, takes = self.directions(board, piece_pos_y, piece_pos_x, move_pos_y,
                                             move_pos_x, move, check_list, 1, -1)
        return move_valid


class Queen(Piece):
    def takes(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, move, move_log):
        queen = Rook(self.color, self.type, self.piece_pos)
        takes_rook = queen.takes(board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, move, move_log)
        queen = Bishop(self.color, self.type, self.piece_pos)
        takes_bishop = queen.takes(board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, move, move_log)
        takes = takes_rook or takes_bishop
        return takes

    def valid_move(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, piece, move, move_log):
        queen = Rook(self.color, self.type, self.piece_pos)
        move_valid_rook = queen.valid_move(board, piece_pos_y, piece_pos_x, move_pos_y,
                                           move_pos_x, piece, move, move_log)
        queen = Bishop(self.color, self.type, self.piece_pos)
        move_valid_bishop = queen.valid_move(board, piece_pos_y, piece_pos_x, move_pos_y,
                                             move_pos_x, piece, move, move_log)
        move_valid = move_valid_rook or move_valid_bishop
        return move_valid


class Knight(Piece):
    def takes(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, move, move_log):
        takes = False
        ab_pos_change_y = abs(move_pos_y - piece_pos_y)
        ab_pos_change_x = abs(move_pos_x - piece_pos_x)
        if self.color_check(move) is True:
            if ab_pos_change_y == 2 and ab_pos_change_x == 1:
                takes = True
            elif ab_pos_change_y == 1 and ab_pos_change_x == 2:
                takes = True
        return takes

    def valid_move(self, board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, piece, move, move_log):
        move_valid = False
        ab_pos_change_y = abs(move_pos_y - piece_pos_y)
        ab_pos_change_x = abs(move_pos_x - piece_pos_x)
        if move == ".." or self.color_check(move) is True:
            if ab_pos_change_y == 2 and ab_pos_change_x == 1:
                move_valid = True
            elif ab_pos_change_y == 1 and ab_pos_change_x == 2:
                move_valid = True
        return move_valid


def highlight_moves(piece_pos, board, move_log):
    board_new = board.copy()
    piece_pos_y, piece_pos_x = notation_to_index(piece_pos)
    piece_notation = board[piece_pos_y, piece_pos_x]
    piece_obj = Piece.create(piece_notation[0], piece_notation[1], piece_pos)
    for i, row in enumerate(board):
        for j, square in enumerate(row):
            move_pos = index_to_notation(i, j)
            move = board[i][j]
            valid = piece_obj.valid_move(board, piece_pos_y, piece_pos_x, i, j, piece_notation, move, move_log)
            if piece_obj.type == "K":
                if valid is False:
                    valid, king_side = piece_obj.castle(board, piece_pos_y, piece_pos_x, i,
                                                        j, piece_notation, move_log)
                else:
                    check = piece_obj.check_move_king(board, piece_pos_y, piece_pos_x, i, j, move_log)
                    valid = not check
            if valid is False and move_pos != piece_pos:
                board_new[i][j] = "XX"
    return board_new


# verify if opposite king in check
def current_check(board, move_pos_y, move_pos_x, move_log):
    piece_selected = board[move_pos_y][move_pos_x]
    for i, row in enumerate(board):
        for j, square in enumerate(row):
            king = board[i, j]
            if piece_selected != ".." and king[0] != piece_selected[0] and king[1] == "K":
                pos = index_to_notation(move_pos_y, move_pos_x)
                piece_obj = Piece.create(piece_selected[0], piece_selected[1], pos)
                if piece_obj.takes(board, move_pos_y, move_pos_x, i, j, king, move_log):
                    return True
    return False


# verify if own king in check
def current_check_king(board, king_y, king_x, move_log):
    king = board[king_y][king_x]
    for i, row in enumerate(board):
        for j, square in enumerate(row):
            piece = board[i, j]
            if piece != ".." and king[0] != piece[0] and king[1] == "K":
                pos = index_to_notation(i, j)
                piece_obj = Piece.create(piece[0], piece[1], pos)
                if piece_obj.takes(board, i, j, king_y, king_x, king, move_log):
                    return True
    return False


# for pins
def self_check(board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, move_log):
    board_copy = board.copy()
    board_copy[move_pos_y][move_pos_x] = board_copy[piece_pos_y][piece_pos_x]
    board_copy[piece_pos_y][piece_pos_x] = ".."
    piece_selected = board_copy[move_pos_y][move_pos_x]
    king, king_y, king_x, king_pos = None, None, None, None
    for i, row in enumerate(board):
        for j, square in enumerate(row):
            iter_piece = board[i, j]
            if piece_selected != ".." and iter_piece[0] == piece_selected[0] and iter_piece[1] == "K":
                king = iter_piece
                king_y, king_x = i, j
                king_pos = board_copy[king_y][king_x]
                break
    if (king, king_y, king_x, king_pos) == (None, None, None, None):
        return False

    for i, row in enumerate(board):
        for j, square in enumerate(row):
            piece = board[i, j]
            if piece != ".." and king[0] != piece[0]:
                pos = index_to_notation(i, j)
                piece_obj = Piece.create(piece[0], piece[1], pos)
                if piece_obj.takes(board_copy, i, j, king_y, king_x, king, move_log):
                    return True
    return False


# moving into check
def future_check(board, piece_pos_y, piece_pos_x, move_pos_y, move_pos_x, move_log):
    board_copy = board.copy()
    board_copy[move_pos_y][move_pos_x] = board_copy[piece_pos_y][piece_pos_x]
    board_copy[piece_pos_y][piece_pos_x] = ".."
    for i, row in enumerate(board):
        for j, square in enumerate(row):
            iter_piece = board[i, j]
            if iter_piece != "..":
                if current_check(board_copy, i, j, move_log):
                    return True
    return False


def checkmate(prev_piece, board, move_log):
    for i, row in enumerate(board):
        for j, square in enumerate(row):
            king = board[i, j]
            if king[0] != prev_piece[0] and king[1] == "K":
                king_pos = index_to_notation(i, j)
                set_1 = {king, "XX"}
                set_2 = {"XX", king}
                for m, row in enumerate(board):
                    for n, square in enumerate(row):
                        piece = board[m, n]
                        if piece != ".." and king[0] == piece[0] and piece[1] != "K":
                            piece_pos = index_to_notation(m, n)
                            piece_obj = Piece.create(piece[0], piece[1], piece_pos)
                            for k, row in enumerate(board):
                                for l, square in enumerate(row):
                                    move = board[k, l]
                                    valid = piece_obj.valid_move(board, m, n, k, l,
                                                                 piece, move, move_log)
                                    if valid is True:
                                        board_copy = board.copy()
                                        board_copy[k][l] = board_copy[m][n]
                                        board_copy[m][n] = ".."
                                        check = current_check_king(board_copy, i, j, move_log)
                                        if check is False:
                                            return False

                board_moves = highlight_moves(king_pos, board, move_log)
                board_set = set(board_moves.flatten())
                if board_set == set_1 or board_set == set_2:
                    return True
                else:
                    return False


def stalemate(board, move_log):
    prev_move_log = board.move_log[-1]
    prev_move = prev_move_log[2]
    prev_piece = prev_move_log[0]
    board_set = set(board.flatten())
    set_1 = {"..", "bK", "wK"}
    set_2 = {"..", "bK", "wK", "wB"}
    set_3 = {"..", "bK", "wK", "wN"}
    set_4 = {"..", "bK", "wK", "bN"}
    set_5 = {"..", "bK", "wK", "bB"}
    sets = [set_1, set_2, set_3, set_4, set_5]
    if board_set in sets:
        return True
    check_list = []
    for i, row in enumerate(board):
        for j, square in enumerate(row):
            piece = board[i, j]
            if piece[0] != prev_piece[0] and piece != "..":
                piece_pos = index_to_notation(i, j)
                board_moves = highlight_moves(piece_pos, board, move_log)
                board_set = set(board_moves.flatten())
                set_1 = {piece, "XX"}
                if board_set == set_1:
                    check_list.append(0)
                else:
                    check_list.append(1)
                    break
    if sum(check_list) == 0:
        return True

    m = len(move_log)
    if m > 5:
        move_1 = move_log[-1]
        move_2 = move_log[-2]
        move_3 = move_log[-3]
        move_4 = move_log[-4]
        move_5 = move_log[-5]
        move_6 = move_log[-6]
        if move_1 == move_5 and move_2 == move_6:
            if move_1[1] == move_3[2] and move_1[2] == move_3[1]:
                if move_2[1] == move_4[2] and move_2[2] == move_4[1]:
                    if move_2[0] == move_4[0] and move_1[0] == move_3[0]:
                        print("Threefold repetition.")
                        return True
    else:
        return False

