def mean(L):
    return sum(L) / len(L)


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
    E = Halton(4, 2)
    print(E)
