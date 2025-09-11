# efficient attack on shift cipher using prob distribution of p_i^2 in plaintext = 0.065
# we find closest to 0.065 for a chosen plaintext p_i x c_(i+j) for all j > 0 in ciphertext
# until we find closest to 0.065 => this is then key so difference between these letters = shift
# need to do all 25 possible shifts for c_i+j until vector p_i x c_i = 0.065
import math

with open("shift_cipher_enc.txt","r") as f:
    ciphertext = f.read()

# alph and list of chars ciphertext
def message_to_list(message_in):
    clean = message_in.translate(str.maketrans({'.': '', ' ': ''})).upper()
    clean_list = [i for i in clean]
    return clean_list


alphabet = "abcdefghijklmnopqrstuvwxyz"
alphabet = list(alphabet.upper())
ciphertext = message_to_list(ciphertext)


# find the freq of each letter in ciphertext
def freq_counter(ref_list, list_find_freqs):
    out_list = []
    for el in ref_list:
        count = 0
        for elem in list_find_freqs:
            if elem == el:
                count = count + 1
        out_list.append(count)
    rel_freqs = [float(i/len(list_find_freqs)) for i in out_list]
    return rel_freqs


def list_shifter(input_list):
    k = []
    s = [i for i in range(len(input_list))]
    for i in s:
        if i >= len(input_list):
            k.append(input_list[-1])
        else:
            k.append(input_list[i - 1])
    return k


frequencies = list(enumerate(freq_counter(alphabet, ciphertext)))
alphabet = list(enumerate(alphabet))


# use english language frequencies to find relative freqs + prove dot prod is roughly 0.065
avg_letter_freqs = [8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.2, 0.8, 4.0, 2.4, 6.7, 1.5, 1.9, 0.1, 6.0, 6.3,
                    9.1, 2.8, 1.0, 2.4, 0.2, 2.0, 0.1]
tot = sum(avg_letter_freqs)
avg_letter_freqs = [i/tot for i in avg_letter_freqs]
alph_avg_freq_zip = list(enumerate(avg_letter_freqs))


def product_finder(vec_1, vec_2, shift_number, list_out):
    listv1_1, listv1_2 = zip(*vec_1)
    listv2_1, listv2_2 = zip(*vec_2)
    prods_list = []
    for i in listv1_1:
        prod = listv2_2[i] * listv1_2[i]
        prods_list.append(prod)
    total = sum(prods_list)
    list_out.append((shift_number, total))


def find_shift():
    p_i = 0.065
    percent_tolerance = 0.05
    p_i_upper = p_i * (1.00 + percent_tolerance)
    p_i_lower = p_i * (1.00 - percent_tolerance)
    final_sq_list = []
    l1 = list_shifter(alph_avg_freq_zip)
    product_finder(frequencies, l1, 1, final_sq_list)
    obj = {}
    for i in range(1, len(alphabet)):
        obj['l' + str(i)] = []
    ab = list(obj.keys())
    ab[0] = l1
    for i in range(1, len(alphabet) - 1):
        ab[i] = ab[i - 1]
        ab[i] = list_shifter(ab[i])
        product_finder(ab[i], frequencies, i + 1, final_sq_list)
    shift_found = 0
    for i in final_sq_list:
        if p_i_lower < i[1] < p_i_upper:
            shift_found = i[0]
    return shift_found


def dec(input_cipher, shift_int):
    indices = []
    for i in input_cipher:
        for x in alphabet:
            if i == x[1]:
                indices.append(x[0])
    unshifted = []
    for i in indices:
        new_out = int(i) - shift_int
        if new_out < 0:
            unshifted.append(len(alphabet) + new_out)
        if new_out == 0:
            unshifted.append(0)
        else:
            unshifted.append(new_out)
    unshifted_letters = []
    for i in unshifted:
        for x in alphabet:
            if i == x[0]:
                unshifted_letters.append(x[1])
    cracked = ''.join(unshifted_letters)
    return cracked


shift = find_shift()
plaintext = dec(ciphertext, shift)
ciphertext = ''.join(ciphertext)
print(shift)
print(ciphertext)
print(plaintext)
