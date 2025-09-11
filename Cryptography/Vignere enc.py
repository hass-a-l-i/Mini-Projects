# Gen, Enc, Dec for Vignere cipher

# Gen = make a key of set length and repeat for length of message
# mathematically - we choose random period according to prob distr over ints (arbitrary)
word_for_key = "plain"

with open("input_vigere_enc.txt","r") as f:
    message = f.read()

alphabet = "abcdefghijklmnopqrstuvwxyz"
alphabet_stored = list(alphabet.upper())
alphabet = list(enumerate(alphabet.upper()))


def modulo(incrementor, modulus_list_length, modulus_list_in):
    factor = int(incrementor / modulus_list_length)
    a = incrementor - (modulus_list_length * factor)
    letter_in = modulus_list_in[a]
    return letter_in


def gen(message_in, word_to_key):
    clean = message_in.translate(
        str.maketrans({'.': '', ' ': '', '"': '', '!': '', '?': '', ',': '', '\'': ''})).upper()
    clean_list = [i for i in clean]
    clean_key = [i.upper() for i in word_to_key]
    key_out = []
    length_list = len(clean_list)
    length_key = len(clean_key)
    for i in range(length_list):
        letter = modulo(i, length_key, clean_key)
        key_out.append(letter)
    return key_out, clean_list


# Enc = use plaintext message m_i => then use key k_i to do m_i + mod(k_t)_t with period t
def enc(key_in, message_in):
    ind_key = []
    for i in key_in:
        for x in alphabet:
            if i == x[1]:
                ind_key.append(x[0])
    clean = message_in.translate(str.maketrans({'.': '', ' ': ''})).upper()
    clean_list = [i for i in clean]
    ind_message = []
    for i in clean_list:
        for x in alphabet:
            if i == x[1]:
                ind_message.append(x[0])
    sum = []
    for i in range(len(ind_message)):
        a = ind_message[i]
        b = ind_key[i]
        c = a + b
        if int(c) > 25:
            c = int(c) - 26
        sum.append(c)
    encr = []
    for i in sum:
        for x in alphabet:
            if i == x[0]:
                encr.append(x[1])
    encr = ''.join(encr)
    return encr


key, message_clean = gen(message, word_for_key)
encrypted = enc(key, message)
message_clean = ''.join(message_clean)


# Dec = reverse of Enc
print(alphabet)
print(key)
print(len(message_clean))
print(len(encrypted))
print(message_clean)
print(encrypted)
with open("vignere_enc.txt", "w") as f:
    f.write(encrypted)