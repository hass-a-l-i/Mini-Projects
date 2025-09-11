# Q:
# Define Gen, Enc, Dec for mono-alphabetic substitution cipher

# Gen = random permutation of i = {0,25} for all letters of alphabet => generates key
import random


def gen(*args):
    set_seed = args
    if not args:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        alphabet_stored = list(alphabet.upper())
        alphabet = list(alphabet.upper())
        from random import shuffle
        shuffle(alphabet)
        new_key = list(zip(alphabet, alphabet_stored))
    else:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        alphabet_stored = list(alphabet.upper())
        alphabet = list(alphabet.upper())
        random.Random(set_seed).shuffle(alphabet)
        new_key = list(zip(alphabet, alphabet_stored))
    return new_key


# Enc = substitute message with key => using plaintext chars m_i (is subset of i ofc i.e alphabet)
# find ciphertext chars c_i = Enc(m_i)
message = "I am pretty sure this is my calling in life. Remember to group and remove full stops to make harder to " \
          "crack. Make converter for message within Enc to make it easier."


def enc(message_in, key_in):
    clean = message_in.translate(str.maketrans({'.': '', ' ': ''})).upper()
    clean_list = [i for i in clean]
    message_enc = []
    for i in clean_list:
        for x in key_in:
            if i == x[0]:
                message_enc.append((x[1]))
    enc_str = ''.join(message_enc)
    return enc_str


key = gen()
print(key)
encrypted = enc(message, key)
print(message)
print(encrypted)
# Dec = inverse of enc => apply function Dec(c_i) = m_i
