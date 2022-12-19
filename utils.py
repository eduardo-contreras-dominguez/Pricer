from numpy import random
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
        u1 = random.uniform()
        u2 = random.uniform()
        x = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        y = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        simulated.append(x)
        simulated.append(y)
    if len(simulated) == number_of_variables:
        output = simulated
    else:
        output = simulated[:-1]
    return output


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
    sim = BoxMuller(10000)
    plt.hist(sim, 20)
    plt.show()
    print(sim)
