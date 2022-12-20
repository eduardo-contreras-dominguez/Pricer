import numpy as np
import math
from matplotlib import pyplot as plt


def mean(L):
    return sum(L) / len(L)


def BoxMuller(number_of_variables):
    # Cuantos pares de variables generamos. Por ejemplo si tenemos 17 variables, generamos 18 y soltamos una.
    simulated = []
    if number_of_variables % 2 == 0:
        number_simulations = int(number_of_variables / 2)
    else:
        number_simulations = int((number_of_variables // 2) + 1)
    for simulation in range(number_simulations):
        u1 = np.random.uniform()
        u2 = np.random.uniform()
        x = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        y = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        simulated.append(x)
        simulated.append(y)
    if len(simulated) == number_of_variables:
        output = simulated
    else:
        output = simulated[:-1]
    return output


def cholesky(cov):
    """
    Calculate the Cholesky decomposition of a correlation matrix.

    Parameters:
    - corr (numpy array): the correlation matrix

    Returns:
    - L (numpy array): the lower-triangular matrix of the Cholesky decomposition
    """
    n = cov.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(cov[i][i] - s)
            else:
                L[i][j] = (cov[i][j] - s) / L[j][j]
    return L


def Halton(n, b):
    """
    This method will compute the nth number of Halton Sequence in base b

    :param n: Number of the sequence
    :param b: Base
    :return: Float number, nth term of Halton's sequence
    """
    n0 = n
    h = 0
    f = 1 / b
    while n0 > 0:
        n1 = n0 // b
        r = n0 % b
        h = h + f * r
        f = f / b
        n0 = n1
    return h


if __name__ == "__main__":
    # E = Halton(4, 2)
    sim = cholesky(np.array([[1, .6], [.6, 1]]))
    print(sim)
