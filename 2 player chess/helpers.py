dict_x = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
dict_y = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}
inv_dict_x = {v: k for k, v in dict_x.items()}
inv_dict_y = {v: k for k, v in dict_y.items()}


def notation_to_index(string_in):
    if len(string_in) != 2:
        return -1, -1
    x = string_in[0]
    y = string_in[1]
    try:
        return dict_y[y], dict_x[x]
    except KeyError:
        return -1, -1


def index_to_notation(y_coord, x_coord):
    letter = str(inv_dict_x[x_coord])
    number = str(inv_dict_y[y_coord])
    ls = [letter, number]
    notation = "".join(ls)
    return notation


