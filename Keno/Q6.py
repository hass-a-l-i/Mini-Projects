import numpy as np

all_numbers = np.arange(1, 81, 1)


def faulty_keno(bias, no_draws, biased_numbers):
    probs = np.ones(80)
    probs = [i / sum(probs) for i in probs]
    unbiased_numbers = [i for i in all_numbers if i not in biased_numbers]

    for i in biased_numbers:
        probs[i - 1] = probs[i - 1] + bias
    for i in unbiased_numbers:
        probs[i - 1] = probs[i - 1] - (len(biased_numbers) * bias) / len(unbiased_numbers)

    probs = np.array(probs)
    probs = probs / probs.sum()

    all_draws = []
    for i in range(no_draws):
        draw = np.random.choice(all_numbers, 20, replace=False, p=probs)
        all_draws.extend(draw)
    return all_draws


def statistical_test(all_draws, no_draws, thresh):
    counts = []
    for i in all_numbers:
        ctr = 0
        for j in all_draws:
            if i == j:
                ctr = ctr + 1
        counts.append(ctr)

    expected = np.full(80, no_draws * 20 / 80)
    residuals = (counts - expected) / np.sqrt(expected)
    biased_numbers_guess = []
    for i in range(len(residuals)):
        if residuals[i] > thresh:
            biased_numbers_guess.append(i+1)
    return biased_numbers_guess


if __name__ == "__main__":
    bias_eg = 0.05 / 100
    no_draws_eg = 100000
    biased_numbers_eg = [5, 14, 39, 71]
    print("Actual biased numbers of faulty Keno:", biased_numbers_eg)
    thresh_eg = 3
    total_draws = faulty_keno(bias_eg, no_draws_eg, biased_numbers_eg)
    guess_eg = statistical_test(total_draws, no_draws_eg, thresh_eg)
    print("Statistical test output biased numbers:",  guess_eg)




