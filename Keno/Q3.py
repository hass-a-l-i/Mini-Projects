from Q2 import results_q2, _2dp, df, r_ls

payoffs = [3, 15, 100, 1000, 25000, 2500000]

results_q3 = []

for i in range(len(r_ls)):
    ev = payoffs[i] * results_q2[i][1]
    results_q3.append((r_ls[i], payoffs[i], results_q2[i][1], ev))

r_ls, payouts, probs, ev_ls = zip(*results_q3)
df = df.drop(columns=["Odds"])
df.insert(1, "P(r)", payouts)
df.insert(3, "EV", ev_ls)
df["EV"] = df["EV"].apply(_2dp)

if __name__ == "__main__":
    print(df.to_string(index=False))
    print()
    print("Expected Value = ", round(sum(ev_ls), 2))

