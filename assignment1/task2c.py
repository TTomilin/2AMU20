import math

import numpy as np

if __name__ == '__main__':
    n = int(5e6)
    x = np.random.uniform(0, 2 * math.pi, size=n)
    y = np.random.uniform(0, 2 * math.pi, size=n)
    K = (2 * math.pi) ** 2
    expectation = np.sin(np.sqrt(x**2 + y**2) + x + y).mean()
    print(K)
    print(expectation)
    print(expectation * K)
