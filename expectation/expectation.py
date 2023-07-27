import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
from pathlib import Path
import matplotlib.pyplot as plt
import os

FILE_PATH = (
    "temperatures.xlsx"  # read excel file containing temperatures for all timezones
)


def f(x, a0, b0, *coefficients) -> pd.Series:
    y_lin = a0 + b0 * x
    if len(coefficients) % 2 != 0:
        raise ValueError("Must provide even number of arguments.")
    y_four = 0
    for i, (a, b) in enumerate(zip(coefficients[::2], coefficients[1::2])):
        n = i + 1
        y_four += a * np.cos(n * 2 * np.pi * x) + b * np.sin(n * 2 * np.pi * x)
    return y_lin + y_four


def expectation_values(y: pd.Series, x: pd.Series, n_coeff: int = 6) -> pd.Series:
    """This function calculates the expectation value at one given climatezone.

    Args:
        y (pd.Series): y defined as the historic temperature at one climatezone
        x (pd.Series): x defined as the datetime index of timeseries
        n_coeff (int, optional): the ideal number of coefficients used to fit the
                expectation curve smoothly. Defaults to 6.

    Returns:
        pd.Series: returns the best fit coefficients, the historic residual and the
                expection value at a particular climatezone
    """

    best_fit_coefficients, *_ = curve_fit(f, x, y, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_fit = f(x, *best_fit_coefficients)
    historic_residuals = np.round((y - y_fit), 4)
    # f_with_best_fit = lambda x: f(x, *best_fit_coefficients)

    return best_fit_coefficients, historic_residuals, y_fit


def expectation() -> pd.DataFrame:
    """This function finds the expectation values of all the climate zone and saves the
    coefficients of the expectation function into an excel file.

    Returns:
        pd.DataFrame: returns the historical deviations after calculating the expectation
                    values of all the climate zones
    """
    t = pd.read_excel(FILE_PATH, index_col=0)
    climatezones = [
        "t_1",
        "t_2",
        "t_3",
        "t_4",
        "t_5",
        "t_6",
        "t_7",
        "t_8",
        "t_9",
        "t_10",
        "t_11",
        "t_12",
        "t_13",
        "t_14",
        "t_15",
    ]

    exp_coeff = pd.DataFrame(index=None)
    deviations = pd.DataFrame(index=None)
    for cz in climatezones:
        y = signal.detrend(t[cz].values, type="linear")
        x = t.index
        (
            best_fit_coefficients,
            historic_deviations,
        ) = expectation_values(y, x)
        exp_coeff = pd.concat([exp_coeff, pd.DataFrame(best_fit_coefficients)], axis=1)
        deviations = pd.concat([deviations, pd.DataFrame(historic_deviations)], axis=1)
    exp_coeff.columns = climatezones
    deviations.columns = climatezones
    deviations.index = t.index
    exp_coeff.to_excel(
        "curvefitting_coefficients.xlsx", sheet_name="curvefitting values"
    )
    return deviations
