# shifting each letter by  set amount

# define functions
def message_to_list(message_in):
    clean = message_in.translate(str.maketrans({'.': '', ' ': ''})).upper()
    clean_list = [i for i in clean]
    return clean_list


# message
message = "I am pretty sure this is my calling in life. Remember to group and remove full stops to make harder to " \
          "crack. Make converter for message within Enc to make it easier. I need a lot of text here so it becomes " \
          "easier to decrypt. Making sure I have at least one whole paragraph. I will become the best codebreaker " \
          "and fulfill my mission in this life. I need a zoo trip."


# generate key = random number between 1 and 25 to add (not 26 as otherwise may not encrypt if mod(26)
def gen(set_shift):
    import random
    if set_shift == 0 or set_shift > 25:
        shift = random.randint(1, 25)
    else:
        shift = set_shift
    return shift


# encrypt message
def enc(message_in, key_in):
    message_in = message_to_list(message_in)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    alphabet = list(enumerate(alphabet.upper()))
    indices_message = []
    for i in message_in:
        for x in alphabet:
            if i == x[1]:
                indices_message.append(x[0])
    shifted_message_ind = []
    for i in indices_message:
        new_in = i + key_in
        if new_in >= len(alphabet):
            shifted_message_ind.append(new_in - len(alphabet))
        else:
            shifted_message_ind.append(new_in)
    encr = []
    for i in shifted_message_ind:
        for x in alphabet:
            if i == x[0]:
                encr.append(x[1])
    encr = ''.join(encr)
    return encr


key = gen(13)
encrypted = enc(message, key)


print(message)
message = message_to_list(message)
message = ''.join(message)
print(key)
print(message)
print(encrypted)

with open("shift_cipher_enc.txt", "w") as f:
    f.write(encrypted)
