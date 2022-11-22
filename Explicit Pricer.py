# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def European_3D_Option_Value(Vol, Risk_Free_Rate, OptionType, Strike, Expiration, EType, NAS):
    """
    This method will compute all different prices on finite difference grid, going 
    from nowadays (last timestep going backwards) to maturity (k = 0).
    
    :param Vol: 
    :param Risk_Free_Rate: 
    :param OptionType: 
    :param Strike: 
    :param Expiration: 
    :param EType: 
    :param NAS: 
    :return: 
    """"""
    :param Vol: Constant Volatility
    :param Risk_Free_Rate: Constant risk free rate
    :param OptionType: European Call or Put
    :param Strike: Strike option price
    :param Expiration: How far is marturity from now
    :param EType: American ("Y") or European Option
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
    Payoff = S.copy()
    if OptionType == "P":
        q = -1
    else:
        q = 1
    # Step 2: Initializing grid values at k = 0 (at expiry). This would be the contract's payoff.
    for i in range(NAS + 1):
        S[i] = i * dS
        V[i, 0] = max(q * (S[i] - Strike), 0)  # Remember that we are going backwards. K=0 corresponds to expiry.
        Payoff[i] = V[i, 0]

    # Now we fill the interior of the grid.
    # Time loop.
    for k in range(1, NTS):
        # Asset Loop
        for i in range(1, NAS):
            Delta = (V[i + 1, k - 1] - V[i - 1, k - 1]) / (2 * dS)
            Gamma = (V[i + 1, k - 1] - 2 * V[i, k - 1] + V[i - 1, k - 1]) / (dS ** 2)
            Theta = -(1 / 2) * Vol ** 2 * S[i] ** 2 * Gamma - Risk_Free_Rate * S[i] * Delta + Risk_Free_Rate * V[
                i, k - 1]
            V[i, k] = V[i, k - 1] - dt * Theta
        if OptionType == "C":
            V[0, k] = 0
        else:
            V[0, k] = Strike * math.exp(-Risk_Free_Rate * k * dt)
        V[NAS, k] = 2 * V[NAS - 1, k] - V[NAS - 2, k]
        if EType == "Y":
            for i in range(NAS + 1):
                V[i, k] = max(Payoff[i], V[i, k])

    return np.around(V, decimals=3)


def European_2D_Option_Value(Vol, Risk_Free_Rate, OptionType, Strike, Expiration, EType, NAS):
    # Step 1: Defining the variables that will be used.
    dS = 2 * Strike / NAS  # How far we will go in terms of asset prices considered (in this case twice the strike).
    dt = .9 / (Vol ** 2 + NAS ** 2)  # Defining the time-step (this choice is due to algorithm stability).
    NTS = int(Expiration / dt) + 1
    dt = Expiration / NTS  # Ensuring that expiration is an integer number of defined timesteps.
    VOld = np.zeros((NAS + 1))
    VNew = VOld.copy()
    S = np.ones((NAS + 1))
    Payoff = S.copy()
    Delta_Final = S.copy()
    Gamma_Final = S.copy()
    Theta_Final = S.copy()
    output_df = pd.DataFrame()

    if OptionType == "P":
        q = -1
    else:
        q = 1
    # Step 2: Initializing grid values at k = 0 (at expiry). This would be the contract's payoff.
    for i in range(NAS + 1):
        # This will be the first column (or even the index column) of out output dataframe.
        S[i] = i * dS
        VOld[i] = max(q * (S[i] - Strike), 0)  # Remember that we are going backwards. K=0 corresponds to expiry.
        # Payoff will be the second column. Next columns will be today's option value and Greeks.
        Payoff[i] = VOld[i]

    # Now we fill the interior of the grid.
    # Time loop.
    for k in range(1, NTS):
        # Asset Loop
        for i in range(1, NAS):
            Delta = (VOld[i + 1] - VOld[i - 1]) / (2 * dS)
            Gamma = (VOld[i + 1] - 2 * VOld[i] + VOld[i - 1]) / (dS ** 2)
            Theta = -(1 / 2) * Vol ** 2 * S[i] ** 2 * Gamma - Risk_Free_Rate * S[i] * Delta + Risk_Free_Rate * VOld[
                i]
            VNew[i] = VOld[i] - dt * Theta
            # Now that the interior of the grid is filled we must fill boundaries. 
        if OptionType == "C":
            VNew[0] = 0
        else:
            VNew[0] = Strike * math.exp(-Risk_Free_Rate * k * dt)
        VNew[NAS] = 2 * VNew[NAS - 1] - VNew[NAS - 2]
        VOld = VNew.copy()
        if EType == "Y":
            for i in range(NAS + 1):
                VOld[i] = max(Payoff[i], VOld[i])
    for i in range(1, NAS):
        # We only save all greeks values when we are at today's date.
        Delta_Final[i] = (VOld[i + 1] - VOld[i - 1]) / (2 * dS)
        Gamma_Final[i] = (VOld[i + 1] - 2 * VOld[i] + VOld[i - 1]) / (dS ** 2)
        Theta_Final[i] = -(1 / 2) * Vol ** 2 * S[i] ** 2 * Gamma_Final[i] - Risk_Free_Rate * S[i] * Delta_Final[i] + \
                         Risk_Free_Rate * VOld[i]
    # For computing greeks on boundaries we will use forward and bacwards differences instead of central one.
    Delta_Final[0] = (VOld[1] - VOld[0]) / dS
    Delta_Final[NAS] = (VOld[NAS] - VOld[NAS - 1]) / dS
    # For very high and almost null values of the stock value is almost lineal.
    Gamma_Final[0] = 0
    Gamma_Final[NAS] = 0
    Theta_Final[0] = Risk_Free_Rate * VOld[0]
    Theta_Final[NAS] = -(1 / 2) * Vol ** 2 * S[NAS] ** 2 * Gamma_Final[i] - Risk_Free_Rate * S[NAS] * Delta_Final[i] + \
                       Risk_Free_Rate * VOld[NAS]
    # Let us fill the output dataframe.
    output_df["Stock Price"] = S
    output_df["Payoff"] = Payoff
    output_df["Option's Value"] = VOld
    output_df["Delta"] = Delta_Final
    output_df["Gamma"] = Gamma_Final
    output_df["Theta"] = Theta_Final
    return output_df


if __name__ == "__main__":
    sigma = 0.2
    ir = .005
    E = 100
    T = 1
    V = European_2D_Option_Value(sigma, ir, "C", E, T, "Y", 20)
    V["Payoff"].plot()
    print(V)
