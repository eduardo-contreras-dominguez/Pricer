import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
import random


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


def Compute_Vanilla_European_Option_Value(S0, risk_free_rate, volatility, Expiry, Strike, Option_Type="C"):
    """
    Will compute the value of an European Vanilla Option using the asset paths on last method. With deterministic
    Interest rate.

    :param S0: Initial asset price
    :param risk_free_rate: spot risk-free rate
    :param volatility: historical vol
    :param Expiry: maturity of contract
    :param Strike: Strike of the option
    :param Option_Type: Call or put

    :return: PV of payoff's average.
    """
    S_paths = simulate_asset_path(S0, risk_free_rate, volatility, 1000, Expiry)
    last_price = S_paths[-1, :]
    if Option_Type == "C":
        q = 1
    else:
        q = -1
    Payoff = [math.exp(-risk_free_rate * Expiry) * max(q * (element - Strike), 0) for element in last_price]
    Delta_Payoff = [math.exp(-risk_free_rate * Expiry) * max(q * (element * (1 + 0.01) - Strike), 0) - math.exp(
        -risk_free_rate * Expiry) * max(q * (element * (1 - 0.01) - Strike), 0) for element in last_price]
    output = sum(Payoff) / len(Payoff)
    Delta = (sum(Delta_Payoff) / len(Delta_Payoff)) / 2 * 0.01 * S0
    return output, Delta


def Compute_Barrier_Option_Value(S0, risk_free_rate, volatility, Expiry, Strike, Barrier, Option_Type="C",
                                 Barrier_Type="I"):
    """
    Will compute the value of an European Vanilla Option using the asset paths on last method. With deterministic
    Interest rate.

    :param S0: Initial asset price
    :param risk_free_rate: spot risk-free rate
    :param volatility: historical vol
    :param Expiry: maturity of contract
    :param Strike: Strike of the option
    :param Barrier: Barrier value
    :param Option_Type: Call or put
    :param Barrier_Type: "I" or "O" for knock-in or knock-out

    :return: PV of payoff's average.
    """
    # First analyze the difference between S0 and the barrier to see if it is an up or down option.
    if S0 < Barrier:
        Directional_Type = "UP"
    else:
        Directional_Type = "DOWN"

    S_paths = simulate_asset_path(S0, risk_free_rate, volatility, 1000, Expiry)
    last_price = S_paths[-1, :]
    if Option_Type == "C":
        q = 1
    else:
        q = -1
    # Option's payoff if this was a vanilla one.
    Payoff = [math.exp(-risk_free_rate * Expiry) * max(q * (element - Strike), 0) for element in last_price]
    #We will see now how many paths touched the barrier during option's life.
    if Barrier_Type == "I":
        if Directional_Type == "UP":
            for path in range(len(last_price)):
                Asset_path = S_paths[:, path]
                if np.count_nonzero(Asset_path >= Barrier) > 0:
                    d = 1
                else:
                    d = 0
                Payoff[path] = d * Payoff[path]
        else:
            for path in range(len(last_price)):
                Asset_path = S_paths[:, path]
                if np.count_nonzero(Asset_path <= Barrier) > 0:
                    d = 1
                else:
                    d = 0
                Payoff[path] = d * Payoff[path]
    if Barrier_Type == "O":
        if Directional_Type == "UP":
            for path in range(len(last_price)):
                Asset_path = S_paths[:, path]
                if np.count_nonzero(Asset_path >= Barrier) > 0:
                    d = 0
                else:
                    d = 1
                Payoff[path] = d * Payoff[path]
        else:
            for path in range(len(last_price)):
                Asset_path = S_paths[:, path]
                if np.count_nonzero(Asset_path <= Barrier) > 0:
                    d = 0
                else:
                    d = 1
                Payoff[path] = d * Payoff[path]

    output = sum(Payoff) / len(Payoff)
    return output


def Compute_Asian_Option_Value(S0, risk_free_rate, volatility, Expiry, LookBack_Period=30, Option_Type="C",
                               DisplayMode=False):
    """
    Will compute the value of an Asian Vanilla Option using the asset paths on last method. With deterministic
    Interest rate.

    :param S0: Initial asset price
    :param risk_free_rate: spot risk-free rate
    :param volatility: historical vol
    :param Expiry: maturity of contract
    :param LookBack_Period: Number of days on which the averaging will be based
    :param Option_Type: Call or put
    :param DisplayMode: A random asset path (risk free rate) will be shown in order to see if everything is correct

    :return: PV of payoff's average.
    """
    # Number of time divisions between today and option's maturity.
    NTS = 1000
    # convert the number of days entered as parameter to number of periods in the simulation.
    # You can choose one year to have 260 days (number of business days on a year). Or 365 if you want
    Number_Periods = int(LookBack_Period * (NTS / (Expiry * 365)))
    S_paths = simulate_asset_path(S0, risk_free_rate, volatility, NTS, Expiry)
    averages = [[S_paths[Number_Periods * i, path] for i in range(1, int(NTS / Number_Periods))] for path in
                range(len(S_paths[0]))]
    averages = list(map(utils.mean, averages))
    last_price = S_paths[-1, :]
    if Option_Type == "C":
        q = 1
    else:
        q = -1
    Payoff = [math.exp(-risk_free_rate * Expiry) * max(q * (last_price[i] - averages[i]), 0) for i in
              range(len(last_price))]
    output = sum(Payoff) / len(Payoff)
    if DisplayMode:
        Path_number = random.randint(0, len(S_paths[0]) - 1)
        path = S_paths[:, Path_number]
        average = averages[Path_number]
        plt.axhline(average, color="red", linestyle="--")
        plt.plot(path)
        plt.show()
    return output


if __name__ == "__main__":
    # price, greek = Compute_Vanilla_European_Option_Value(S0=111, risk_free_rate=0.05, volatility=0.2, Expiry=1,
    # Strike=100)
    # Compute_Barrier_Option_Value(
    #     S0=100,
    #     risk_free_rate=0.05,
    #     volatility=0.2,
    #     Expiry=1,
    #     Strike=110,
    #     Barrier=130,
    #     Option_Type="C",
    #     Barrier_Type="O"
    # )
    Compute_Asian_Option_Value(S0=100, risk_free_rate=0.05, volatility=0.2, Expiry=1, LookBack_Period=30,
                               Option_Type="C")
    
