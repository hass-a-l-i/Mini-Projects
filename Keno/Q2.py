import numpy as np
import pandas as pd
import math


def nCr(n, r):
    return math.factorial(n)/(math.factorial(r) * (math.factorial(n - r)))


r_ls = np.arange(5, 11, 1)

N = 80
R = 20

results_q2 = []
for r in r_ls:
    p = (nCr(10, r) * nCr(70, 20 - r)) / nCr(80, 20)
    odds = 1/p
    results_q2.append((r, p, odds))


def _3sf(a):
    return f"{a:.3g}"


def _2dp(a):
    return f"{a:.2f}"


df = pd.DataFrame(results_q2, columns=["r", "Pr[X = r]", "Odds"])
df["Pr[X = r]"] = df["Pr[X = r]"].apply(_3sf)
df["Odds"] = df["Odds"].apply(_2dp)

if __name__ == "__main__":
    print(df.to_string(index=False))

