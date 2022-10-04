import math
import matplotlib.pyplot as plt
from typing import List, Tuple

import numpy as np


def get_total_prob(omega: np.ndarray, M: float, lamb: float) -> Tuple[float, List[float]]:
    print(f'M={M}')
    cumulative_prob = 0
    p_vals = []
    for k in omega:
        x = M * lamb**k * math.e**(-lamb) / math.factorial(k)
        cumulative_prob += x
        p_vals.append(x)
        print(f'k={k}: x={x:.4f}')
    print(f'Total Probability: {cumulative_prob}\n')
    return cumulative_prob, p_vals


def calc_expected_value(values, weights):
    values = np.asarray(values)
    weights = np.asarray(weights)
    return (values * weights).sum() / weights.sum()


if __name__ == '__main__':
    # Task 2a1
    lambda_ = 4
    sample_space = np.arange(5)
    precision = 5
    M = 1
    total_prob, _ = get_total_prob(sample_space, M, lambda_)
    M = 1 / total_prob
    total_prob, p_vals = get_total_prob(sample_space, M, lambda_)
    assert np.around(total_prob, decimals=precision) == 1, f'Failed to approximate M={M}, total probability: {total_prob}'
    print(f'Final M={M}')

    # Task 2a2
    expected_value = calc_expected_value(sample_space, p_vals)

    # Task 2a3
    experiment_n = np.arange(100, 10000, 100)
    expected_vals = []
    for n in experiment_n:
        dist = np.random.multinomial(n, p_vals)
        expected = (dist * sample_space).sum() / n
        expected_vals.append(expected)
        print(f'Expected value for n={n}: {expected:.4f}')

    # Plot expected_vals vs experiment_n with matplotlib
    plt.style.use('seaborn')
    plt.plot(experiment_n, expected_vals)
    plt.plot(experiment_n, [expected_value] * len(experiment_n), color='red')
    plt.xlabel('Experiments')
    plt.ylabel('Expected value')
    plt.show()



