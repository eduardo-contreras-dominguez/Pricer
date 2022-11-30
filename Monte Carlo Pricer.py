import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


def simulate_asset_path(S0, risk_free_rate, volatility, NTS, T, N=10000):
    """
    Simulating realizations of log-normal risk-neutral random walk

    :param S0: Initial asset price
    :param risk_free_rate: risk-free spot rate
    :param volatility: historical vol
    :param NTS: number of time steps
    :param T: Derivatives expiration
    :param N: Number of realizations

    :return:  2D array (NTS x N) having asset prices for every simulation.
    """
    realization_array = np.zeros((NTS, N))
    # We will start at S0 for every simulation
    realization_array[0, :] = [S0 for simulation in range(N)]
    dt = T / NTS
    for simulation in tqdm(range(N)):
        random_variable = np.random.normal(0, 1, NTS)
        for timestep in range(1, NTS):
            realization_array[timestep, simulation] = realization_array[timestep - 1, simulation] \
                                                      * math.exp((risk_free_rate - 1 / 2 * volatility ** 2) * dt
                                                                 + volatility * math.sqrt(dt) * random_variable[
                                                                     timestep])
    return realization_array

def Compute_Vanilla_European_Option_Value(S0, risk_free_rate, volatility, Expiry, Strike, Option_Type = "C"):
    """
    Will compute the value of an European Vanilla Option using the asset paths on last method.

    :param S0: Initial asset price
    :param risk_free_rate: spot risk-free rate
    :param volatility: historical vol
    :param Expiry: maturity of contract
    :param Strike: Strike of the option
    :param Option_Type: Call or put

    :return: PV of payoff's average.
    """


if __name__ == "__main__":
    df = simulate_asset_path(S0=100, risk_free_rate=0.05, volatility=0.1, NTS=1000, T=1)
    #plt.plot(df)
    #plt.show()
    print("Hello")
