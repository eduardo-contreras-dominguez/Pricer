import utils
import numpy as np
import math


def numerical_euro_call(NDim, NoPts, IntRate, Div, Cov, Asset, Strike, Expiry):
    """

    :param NDim: Number of Assets to be considered
    :param NoPts: Number of Monte Carlo simulations
    :param IntRate: Interest Rate
    :param Div: Dividend Yield for every asset
    :param Cov: Covariance Matrix for every asset.
    :param Asset: Current Asset Price
    :param Expiry: Time until Expiry

    :return: European Call Price
    """
    # We will create the covariance matrix of normal random variables we have to simulate.
    Normal_Cov = np.zeros((NDim, NDim))
    for i in range(NDim):
        for j in range(NDim):
            if i == j:
                Normal_Cov[i][j] = 1
            else:
                Normal_Cov[i][j] = Cov[i][j]
    discount = math.exp(-IntRate * Expiry) / NoPts
    suma = 0
    S = [0 for i in range(NDim)]
    M = utils.cholesky(Cov)
    for k in range(NoPts):
        x = np.dot(M, utils.BoxMuller(NDim))
        for i in range(NDim):
            S[i] = Asset[i] * math.exp((IntRate - Div[i] - 1 / 2 * Cov[i][i]) * Expiry + math.sqrt(Cov[i][i])) * x[
                i] * math.sqrt(Expiry)
        # We suppose a call on a basket of assets (equal weight on each asset)
        Payoff = max(1 / NDim * sum(S) - Strike, 0)
        suma+= Payoff
    return suma*a
