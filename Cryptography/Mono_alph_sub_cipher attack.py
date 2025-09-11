# decrypting mono-alphabetic substitution cipher below
# each letter shifted by a set number for all letters
# use prob distribution of normal english to find common frequencies to decrypt
import operator

input_cipher = "JGRMQOYGHMVBJWRWQFPWHGFFDQGFPFZRKBEEBJIZQQOCIBZKLFAFGQVFZFWWEOGWOPFGFHWOLPHLRLOLFDMFGQWBLWBWQOLKFWB" \
               "YLBLYLFSFLJGRMQBOLWJVFPFWQVHQWFFPQOQVFPQOCFPOGFWFJIGFQVHLHLROQVFGWJVFPFOLFHGQVQVFILEOGQILHQFQGIQVVO" \
               "SFAFGBWQVHQWIJVWJVFPFWHGFIWIHZZRQGBABHZQOCGFHX"
#with open("vignere_enc.txt","r") as f:
 #   input_cipher = f.read()

# 1. convert cipher into list of chars, and do same for alphabet
input_cipher = list(input_cipher)

alphabet = "abcdefghijklmnopqrstuvwxyz"
alphabet = alphabet.upper()
alphabet = list(alphabet)

# 2. import and clean frequencies, then pair with alphabet
freqs = []
with open("Q1_letter_freqs.txt") as txt:
    lines = txt.readlines()
    for x in lines:
        freqs.append(x)

freqs = [elem[9:14] for elem in freqs]
freqs = [float(elem) for elem in freqs]

letter_freq = list(zip(alphabet, freqs))

# 3. for cipher, find freqs of each letter
total = len(input_cipher)


def freq_counter(ref_list, list_find_freqs):
    out_list = []
    for el in ref_list:
        count = 0
        for elem in list_find_freqs:
            if elem == el:
                count = count + 1
        out_list.append(count)
    return out_list


new_freqs = freq_counter(alphabet, input_cipher)
new_freqs = [float(e / total) * 100 for e in new_freqs]
cipher_freqs = list(zip(alphabet, new_freqs))

# 4. find key by matching ascending freqs of both textspaces in new list
letter_freq = sorted(letter_freq, key=operator.itemgetter(1), reverse=True)
cipher_freqs = sorted(cipher_freqs, key=operator.itemgetter(1), reverse=True)

a = []
b = []
i = 0
for x in letter_freq:
    a.append(letter_freq[i][0])
    b.append(cipher_freqs[i][0])
    i = i + 1

key = list(zip(a, b))

# 5. substitute the largest 2 letter freq into cipher

output_cipher = []
for i in input_cipher:
    if i == key[0][1]:
        a = key[0][0]
        output_cipher.append(a)
    elif i == key[1][1]:
        b = key[1][0]
        output_cipher.append(b)
    elif i != (key[0][1] or key[1][1]):
        output_cipher.append(" ")

# 6. use common patterns to solve - e.g. put H between all Ts and Es

output_cipher2 = []
for i in enumerate(output_cipher):
    if output_cipher[i[0] - 1] == 'T' and output_cipher[i[0] + 1] == 'E':
        output_cipher2.append('H')
    else:
        output_cipher2.append(i[1])

# 7. now guess using common patters and words - first funcs - also append key_actual as we go

key_actual = []


def list_to_string(input_str):
    input_str = ''.join(input_str)
    print(input_str)


def next_guess(last_guess, original_cypher, char_old, char_new):
    print(char_old, " is now ", char_new)
    key_actual.append((char_old, char_new))
    new_guess = []
    for i in enumerate(original_cypher):
        if input_cipher[i[0]] == char_old:
            new_guess.append(char_new)
        else:
            new_guess.append(last_guess[i[0]])
    list_to_string(new_guess)
    return new_guess

list_to_string(input_cipher)
# now guesses

# can see Hs are either V or W or G, V most common so assume V = H
output_cipher3 = next_guess(output_cipher2, input_cipher, 'V', 'H')

# assume H is A using TH*T string found later in plaintext so far
output_cipher4 = next_guess(output_cipher3, input_cipher, 'H', 'A')

# found T* before a THE so assume O is O to make TO as no other possible
output_cipher5 = next_guess(output_cipher4, input_cipher, 'O', 'O')

# found OTHE* so G must be R to make OTHER
output_cipher6 = next_guess(output_cipher5, input_cipher, 'G', 'R')

# found THA*, already have T so try THAN meaning L is N
output_cipher7 = next_guess(output_cipher6, input_cipher, 'L', 'N')

# found HO*E*ER - HOWEVER, SO A is V and S IS W
output_cipher8 = next_guess(output_cipher7, input_cipher, 'A', 'V')
output_cipher9 = next_guess(output_cipher8, input_cipher, 'S', 'W')

# TR*TH - then word HOWEVER, so I must be U to make TRUTH
output_cipher10 = next_guess(output_cipher9, input_cipher, 'I', 'U')

# UN*ORTUNATE - E is F
output_cipher11 = next_guess(output_cipher10, input_cipher, 'E', 'F')

# TRUTH HOWEVER ** THAT => missing clearly IS so B is I, W is S
output_cipher12 = next_guess(output_cipher11, input_cipher, 'B', 'I')
output_cipher13 = next_guess(output_cipher12, input_cipher, 'W', 'S')

# NEVERTHE*ESS => Z is L and SO*E => P is M
output_cipher14 = next_guess(output_cipher13, input_cipher, 'Z', 'L')
output_cipher15 = next_guess(output_cipher14, input_cipher, 'P', 'M')

# *E MORE => C is B, SE*URE => J is C
output_cipher16 = next_guess(output_cipher15, input_cipher, 'J', 'C')
output_cipher17 = next_guess(output_cipher16, input_cipher, 'C', 'B')

# USUALL* => R is Y
output_cipher18 = next_guess(output_cipher17, input_cipher, 'R', 'Y')

# *IFFICULT => K IS D
output_cipher19 = next_guess(output_cipher18, input_cipher, 'K', 'D')

# BREA* => X is K, E*TREMELY => D is X
output_cipher20 = next_guess(output_cipher19, input_cipher, 'X', 'K')
output_cipher21 = next_guess(output_cipher20, input_cipher, 'D', 'X')

# E*PERTS => M is P
output_cipher22 = next_guess(output_cipher21, input_cipher, 'M', 'P')

# CRYPTO*RAPHIC => Y is G
output_cipher23 = next_guess(output_cipher22, input_cipher, 'Y', 'G')

# add in the first two most freq from beginning to key
key_actual.append((key[0][1], key[0][0]))
key_actual.append((key[1][1], key[1][0]))
print(key_actual)

# check if key actual returns us what we want
check = []

for i in input_cipher:
    for x in key_actual:
        if i == x[0]:
            a = x[1]
            check.append(a)

list_to_string(check)
print(check == output_cipher23)

"""

for elem in input_cipher:
    elem = ord(elem) + diff
    if elem > ord('Z'):
        elem = elem - 26
    decrypted.append(chr(elem))


max_english = letter_freq[0][0]
max_cipher = cipher_freqs[0][0]

print(letter_freq)
print(cipher_freqs)

diff = 19

indices = []

for i in range(26):
    i = i + 1
    indices.append(i)

mod_list = list(zip(alphabet, indices))
print(mod_list)



for i in input_cipher:
    index = 0
    for x in key:
        if i == x[1]:
            a = x[0]
            output_cipher.append(a)
            index += 1
        if i != x[1]:
            index += 1


"""
