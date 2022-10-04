from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # Task 2a3
    expected_value = 4/3
    experiment_n = np.arange(100, 10000, 100)
    expected_vals = []
    for n in experiment_n:
        X = np.random.triangular(0, 2, 2, size=n)
        expected = np.mean(X)
        expected_vals.append(expected)
        print(f'n={n}, E(X)={expected:.4f}')

    # Plot expected_vals vs experiment_n with matplotlib
    plt.style.use('seaborn')
    plt.plot(experiment_n, expected_vals)
    plt.plot(experiment_n, [expected_value] * len(experiment_n), color='red')
    plt.xlabel('Experiments')
    plt.ylabel('Expected value')
    plt.show()
