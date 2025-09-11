# find period algo
# then run through the shift cipher algo

with open("vignere_enc.txt","r") as f:
    ciphertext = f.read()


# assume t known
def message_to_list(message_in):
    clean = message_in.translate(
        str.maketrans({'.': '', ' ': '', '"': '', '!': '', '?': '', ',': '', '\'': ''})).upper()
    clean_list = [i for i in clean]
    return clean_list


print(ciphertext)
ciphertext = message_to_list(ciphertext)
print(ciphertext)

# long af to code - just find each sub group then find shift then find word used for shift easy


