import matplotlib.pyplot as plt
import pandas
from collections import Counter
import itertools

# database of words opened
words = open("words.txt", "r")
raw_words = words.readlines()

clean = map(lambda s: s.strip(), raw_words)

# map words into a list
clean = list(clean)


# define function for splitting string into list of characters
def split(word):
    return [char for char in word]


lists_chars = []
# split the list of words into list of lists of characters
for x in clean:
    z = split(x)
    data = lists_chars.append(z)

# list comprehension to concatenate list of lists
data = [item for sublist in lists_chars for item in sublist]


# histogram for character vs frequency
def histdistr():
    letter_counts = Counter(data)
    df = pandas.DataFrame.from_dict(letter_counts, orient='index')
    df.plot(kind='bar')
    plt.show()


# create dictionary to count no characters in data set
count = {}
for s in data:
    if s in count:
        count[s] += 1
    else:
        count[s] = 1

char = []
char_freq = []

# use keys for each entry in count dictionary to convert the dictionary into two lists, as long as len(char) > thresh
thresh = 0
for key in count:
    if count[key] > thresh:
        # print(key, count[key])
        char.append(key)
        char_freq.append(count[key])

# make list of tuples for (character, character frequency)
tuple_chars = list(zip(char, char_freq))


# convert list of characters back into string
def convert(s):
    new = ""

    for x in s:
        new += x

    return new


char = convert(char)

# go through every permutation of 5 letter words from pool of chars above defined thresh
list_perms = list(itertools.permutations(char, 5))

list_strings = []

# convert the permutations of chars into strings
for item in list_perms:
    item = convert(item)
    list_strings.append(item)

# find actual words from random permutations using original list of words (overlap)
actual_words = list(set(list_strings) & set(clean))

# make list of chars from words found in overlap
overlap_chars = [list(item) for item in actual_words]

length = len(overlap_chars)
index = list(range(0, length - 1))

most_likely = []

# loop through list of tuples to find any matches to the list of chars found earlier, if match then sum up the second
# half of tuple to find cumulative frequency

for i in index:
    iterator = overlap_chars[i]
    freq = []

    for item in tuple_chars:
        if item[0] in iterator:
            freq.append(item[1])
    # print(kl)
    # print(sum(kl))
    most_likely.append(sum(freq))

# make list of tuples from most likely words and cumulative frequencies
raw_list = list(zip(actual_words, most_likely))

# sort list to go in descending order, finding best to worst guess
sort = sorted(raw_list, key=lambda x: x[1], reverse=True)

# permutations containing same letter redundant so these removed with empty set filter
empty = set()
final_list = []
for a, b in sort:
    if not b in empty:
        empty.add(b)
        final_list.append((a, b))

# write output to txt
with open('final.txt', 'w') as fp:
    fp.write('\n'.join('%s %s' % x for x in final_list))
