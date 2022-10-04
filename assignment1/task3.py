import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    exact_prob = 1/4 * 1/3

    characters = ['F', 'U', 'L', 'L']
    word = ''.join(characters)
    matches = []

    experiment_n = np.arange(100, 10000, 100)
    for n in experiment_n:
        match = 0
        for _ in range(n):
            permutation = np.random.permutation(characters)
            if ''.join(permutation) == word:
                match += 1
        matches.append(match / n)
        print(f'Probability of matching word for n={n}: {match / n:.4f}')
    # Plot expected_vals vs experiment_n with matplotlib
    plt.style.use('seaborn')
    plt.plot(experiment_n, matches)
    plt.plot(experiment_n, [exact_prob] * len(experiment_n), color='red')
    plt.xlabel('Experiments')
    plt.ylabel('Probability')
    plt.show()
