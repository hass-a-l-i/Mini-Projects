import numpy as np
import pandas as pd

draws = np.arange(7, 20, 1)
rev_draws = draws[::-1]
matches = np.arange(7, 10, 1)
rev_matches = matches[::-1]

payoffs = [(20, 5, 3), (20, 6, 15), (20, 7, 100), (20, 8, 1000), (20, 9, 25000), (20, 10, 2500000)]


def a(i, j):
    for item in payoffs:
        if i == item[0] and j == item[1]:
            return item[2]
        elif j == 10:
            return 2500000


for i in rev_draws:
    for j in rev_matches:
        p_match = (10 - j)/(80 - i)
        p_miss = ((70 - (i - j))/(80 - i))
        a_match = a(i + 1, j + 1)
        a_miss = a(i + 1, j)
        a_new = (a_match * p_match) + (a_miss * p_miss)
        payoffs.append((i, j, a_new))

table = payoffs[6:]
table = table[::-1]

df = pd.DataFrame(table, columns=["draws (i)", "matches (j)", "a(i,j)"])
df = df.round(2)

if __name__ == "__main__":
    print(df.to_string(index=False))
