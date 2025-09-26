import math
import numpy as np
import pandas as pd

from Q3 import probs

C = 20 * (10 ** 6)
L = 2.5 * (10 ** 6)
N = 10 ** 8
p = probs[-1]
EV_K = N * p


def poisson(lambda_, k):
    return (np.exp(-lambda_) * (lambda_ ** k)) / math.factorial(k)


k_ls = np.arange(1, 30, 1)
q_ls = []
table = []
for k in k_ls:
    q = poisson(EV_K, k)
    q_ls.append(q)
    table.append((k, round(sum(q_ls), 4)))

df = pd.DataFrame(table, columns=["k", "Pr[K <= K]"])


def find_q(ls, quantile):
    for i, j in ls:
        if j >= quantile:
            return i


q = 0.9999
min_k = find_q(table, q)

m = ((min_k * L) - C) / N

if __name__ == "__main__":
    print("Expected number of wins :", round(EV_K))
    print(df.to_string(index=False))
    print("Minimum number of wins before arriving in", q * 100, "% quantile :", min_k)
    print("Recommended insurance premium :", m)

