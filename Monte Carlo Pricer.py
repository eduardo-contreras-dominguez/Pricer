import numpy as np
import pandas as pd
import math


def simulate_asset_path(S0, risk_free_rate, volatility, NTS, T, N=1000):
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
    for simulation in range(N):
        random_variable = np.random.normal(0, 1, NTS)
        for timestep in range(1, NTS):
            realization_array[timestep, simulation] = realization_array[timestep - 1, simulation] \
                                                      * math.exp((risk_free_rate - 1 / 2 * volatility ** 2) * dt
                                                                 + volatility * math.sqrt(dt) * random_variable[
                                                                     timestep])
    return realization_array


if __name__ == "__main__":
    df = simulate_asset_path(S0=100, risk_free_rate=0.05, volatility=0.2, NTS=1000, T=1)
    print("Hello")
