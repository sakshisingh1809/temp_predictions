import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import statsmodels.api as sm


def partial_autocorrelation(deviations: pd.DataFrame, lags: int = 1) -> pd.Series:
    """This function is used to calculate the partial autocorrelation function

    Args:
        deviations (pd.DataFrame): historic deviations
        lags (int, optional): number of lags to return autocorrelation for.
            Defaults to 15.

    Returns:
        pd.Series: returns the partial autocorrelations for lags 0, 1, …, lags
    """
    return sm.tsa.pacf(deviations, nlags=lags)


def autocorrelation(deviations: pd.DataFrame, lags: int = 15) -> pd.Series:
    """This function is used to calculate the autocorrelation function

    Args:
        deviations (pd.DataFrame): historic deviations
        lags (int, optional): number of lags to return autocorrelation for.
            Defaults to 15.

    Returns:
        pd.Series: returns the autocorrelations for lags 0, 1, …, lags
    """
    # plot_acf(deviations, lags=lags, color="g", title=f"{cz}: autocorrelation")
    return sm.tsa.stattools.acf(deviations, nlags=lags)


def curvefitting(deviations: pd.Series, coeff: pd.Series, lags: int = 15) -> pd.Series:
    coeff = coeff[0 : lags + 1]
    curve = 0
    for i in range(lags):
        curve += coeff[i + 1] * deviations.iloc[lags - i :]
        # print(coeff[i + 1], ",,", deviations.iloc[lags - i :], curve)
    return curve


def pacf_simulation(
    simulated_residuals: pd.Series, pacf_coeff: pd.Series, lags: int = 1
) -> pd.Series:
    pacf_coeff = pacf_coeff[1 : lags + 1]
    simulated_deviations = []
    for i, res in enumerate(simulated_residuals):
        prev_dev = simulated_deviations[i - lags : i]
        current_dev = current_devation(res, prev_dev, pacf_coeff)
        simulated_deviations.append(current_dev)
    return simulated_deviations


def current_devation(
    current_residual: float, prev_deviations: list, pacf_coefficients: list
) -> float:
    shorten_to = min(len(pacf_coefficients), len(prev_deviations))
    pacf_coefficients = pacf_coefficients[:shorten_to]
    prev_deviations = prev_deviations[-shorten_to:]
    return current_residual + sum(
        [coeff * val for coeff, val in zip(pacf_coefficients, prev_deviations[::-1])]
    )
