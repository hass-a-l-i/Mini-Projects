import numpy as np

c1 = np.array(['W', 'B'])
c2 = np.array(['B', 'B'])
c3 = np.array(['W', 'W'])

card_dict = {'c1': c1, 'c2': c2, 'c3': c3}

def card_picker():
    shuffle = np.random.randint(1, 4, 1)
    prob_side = np.random.randint(0, 2, 1)
    for i in range(1, 4):
        if shuffle == i:
            if prob_side == 0:
                #print('c%d' % i)
                #print("Top", card_dict['c%d' % i][0])
                return 'c%d' % i, card_dict['c%d' % i][0]
            else:
                #print('c%d' % i)
                #print("Bottom", card_dict['c%d' % i][1])
                return 'c%d' % i, card_dict['c%d' % i][1]


sides = []
for i in range(0, 10000):
    side = card_picker()
    sides.append(side)

# exclude c3 as no chance of getting white on other side
sides_no_c3 = [item for item in sides if item[0] != 'c3']

# only c1's can have whites on other side
black_sides = [item for item in sides if item[1] == 'B']

whites_given_blacks = [item for item in black_sides if item[0] == 'c1']

print("Prob(x2 = W | x1 = B) = ", len(whites_given_blacks)/ len(black_sides))
