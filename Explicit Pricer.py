# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import math


def European_Option_Value(Vol, Risk_Free_Rate, OptionType, Strike, Expiration, NAS):
    """

    :param Vol: Constant Volatility
    :param Risk_Free_Rate: Constant risk free rate
    :param OptionType: European Call or Put
    :param Strike: Strike option price
    :param Expiration: How far is marturity from now
    :param NAS: Number of asset time steps

    :return: Will return an array representing the grid of explicit finite-difference method.
    """

    # Step 1: Defining the variables that will be used.
    dS = 2 * Strike / NAS  # How far we will go in terms of asset prices considered (in this case twice the strike).
    dt = .9 / (Vol ** 2 + NAS ** 2)  # Defining the time-step (this choice is due to algorithm stability).
    NTS = int(Expiration / dt) + 1
    dt = Expiration / NTS  # Ensuring that expiration is an integer number of defined timesteps.
    V = np.zeros((NAS + 1, NTS + 1))
    S = np.ones((NAS + 1))
    if OptionType == "P":
        q = -1
    else:
        q = 1
    # Step 2: Initializing grid values at k = 0 (at expiry). This would be the contract's payoff.
    for i in range(NAS + 1):
        S[i] = i * dS
        V[i, 0] = max(q * (S[i] - Strike), 0)  # Remember that we are going backwards. K=0 corresponds to expiry.

    # Now we fill the interior of the grid.
    # Time loop.
    for k in range(1, NTS):
        # Asset Loop
        for i in range(1, NAS):
            Delta = (V[i + 1, k - 1] - V[i - 1, k - 1]) / (2 * dS)
            Gamma = (V[i + 1, k - 1] - 2 * V[i, k - 1] + V[i - 1, k - 1]) / (dS ** 2)
            Theta = -(1 / 2) * Vol ** 2 * S[i] ** 2 * Gamma - Risk_Free_Rate * S[i] * Delta + Risk_Free_Rate * V[i, k - 1]
            V[i, k] = V[i, k - 1] - dt * Theta
        if OptionType == "C":
            V[0, k] = 0
        else:
            V[0, k] = Strike * math.exp(-Risk_Free_Rate * k * dt)
        V[NAS, k] = 2 * V[NAS - 1, k] - V[NAS - 2, k]
    return np.around(V, decimals=3)


if __name__ == "__main__":
    sigma = 0.2
    ir = .005
    E = 100
    T = 1
    V = European_Option_Value(sigma, ir, "C", E, T, 20)
    print(V)
