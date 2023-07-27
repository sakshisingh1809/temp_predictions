import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from expectation import expectation
from visualise import single_monthly_plot, plot_long_graph
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from PACF import PACF
from PCA import PCA

FILE_PATH = "temperatures.xlsx"  # excel file containing temperatures for all timezones
PACF_FILE_PATH = "pacf_coefficients.xlsx"
temp = {}


def model_A(x: pd.Series, y: pd.Series) -> pd.Series:
    """This model outputs the expectation value of the temperature at each
    climatezone so as to obtain a smooth-looking curve

    Args:
        x (pd.Series): index of timeseries
        y (pd.Series): historical temperature of one climatezone

    Returns:
        pd.Series: returns the historical deviations/ offsets after calculating
            the expectation values of the timeseries
    """
    (
        best_fit_coefficients,
        historic_deviations,
        yfit,
    ) = expectation.expectation_values(y, x)

    return historic_deviations, yfit


def model_B(tau: pd.Series, deviations: pd.Series, lag: int) -> pd.Series:
    """This model outputs the partial autocorrelation values at climatezones
    so as to obtain a more realistic-looking curve.

    Args:
        tau (pd.Series): index of timeseries
        deviations (pd.Series): historic offsets from the expectation value
                        from model A
        lag (int): the correlation between values that are lag time periods apart
                in the timeseries

    Returns:
        pd.Series: returns the simulated future temperature offset for each
            climatezones
    """
    acf_coeff = PACF.autocorrelation(deviations, lag + 1)
    acf_deviations = PACF.curvefitting(deviations, acf_coeff, lag)
    return acf_deviations


def fitting_model(
    tau: pd.Series, hist_tmp: pd.Series, cz: pd.Series, lag: int
) -> pd.Series:
    """This function fits the historic temperatures to the two models A and B,
    which calculates the expectation values of temperatures and simulated
    offsets values of temperatures respectively.

    Args:
        tau (pd.Series): the index of timeseries (converted to floating
                    numbers centered around 2000)
        hist_tmp (pd.Series): the original historic temperature of one
                    climatezone
        cz (pd.Series): one climatezone
        lag (int): the correlation between values that are lag time periods apart
                in the timeseries
    Returns:
        pd.Series: the residual after applying the model B
    """

    deviations, yfit = model_A(tau, hist_tmp)
    acf_deviations = model_B(tau, pd.Series(deviations), lag)
    epsilon_acf = deviations[lag:] - acf_deviations

    return epsilon_acf, yfit


def using_model(
    tau: pd.Series,
    y: pd.Series,
    deviations: pd.Series,
    cz: pd.Series,
    yfit: pd.Series,
    lag: int,
) -> pd.Series:
    """This function uses the the simulated residuals from model B to generate
    the simulated temperatures for the given timeseries in one climatezone.

    Args:
        tau (pd.Series): index of timeseries
        y (pd.Series): the original historic temperature of one
                    climatezone
        deviations (pd.Series): the simulated future temperature offset from model B
        cz (pd.Series): one climatezone
        yfit (pd.Series): the expectation value
        lag (int): the correlation between values that are lag time periods apart
                in the timeseries

    Returns:
        pd.Series: the simulated temperatures obtained by adding the simulated
        deviations to the expectation values
    """
    mu = deviations.mean()
    sigma = deviations.std()
    simulated_residuals = np.random.normal(mu, sigma, size=tau.size)

    pacf_coeff = pd.read_excel(PACF_FILE_PATH, index_col=0)
    simulated_deviations = PACF.pacf_simulation(
        simulated_residuals, pacf_coeff[cz], lag
    )
    simulated_temps = yfit + simulated_deviations
    # plot_simulations(tau, y, simulated_temps, lag)

    """ sim = pd.DataFrame(
        {
            "deviation": np.sort(deviations),
            "simulated_deviation": np.sort(simulated_deviations[lag - 1 :]),
        },
        np.linspace(0, 1, len(deviations)),
    )
    sim.plot() """
    return simulated_deviations, simulated_temps


def plot_simulations(x: pd.Series, y: pd.Series, simulated_temps: pd.Series, lags: int):
    "Function to plot the actual temperatures vs the simulted temperatures"
    x = x[lags:]
    y = y[lags:]
    year = 2
    days = -1 * 365 * year
    pd.DataFrame(
        {
            "Actual temperatures": y[days:],
            "Simulated temperatures": simulated_temps[days:],
        },
        x[days:],
    ).plot()
    plt.show()


def model(x: pd.Series, y: pd.Series, cz: str, lag: int) -> pd.Series:
    """This function evaluates the two models that are used to first calculate the
    expectation value of the historic deviations (more like a smooth curve after fitting
    the expectations/ means) and second calculates the partial autocorrelation between
    climatezones so as to obtain a more realistic looking curve.

    This function encapsulates the two aspects: fitting the model and using the model.

    Args:
        x (pd.Series): index of timeseries
        y (pd.Series): the original historic temperature of one
                    climatezone
        cz (pd.Series): one climatezone
        lag (int): the correlation between values that are lag time periods apart
                in the timeseries

    Returns:
        pd.Series: returns the simulated temperatures for a given climatezone
    """
    deviations, yfit = fitting_model(x, y, cz, lag)
    simulated_deviations, simulated_temps = using_model(
        x, y, pd.Series(deviations), cz, yfit, lag + 1
    )
    temperatures = pd.DataFrame(
        {
            "tau": x[lag:],
            "actual_temps": y[lag:],
            "deviations": deviations,
            "simulated_deviations": simulated_deviations[lag:],
            "expected_temps": yfit[lag:],
            "simulated_temps": simulated_temps[lag:],
        }
    )
    """ sim = pd.DataFrame(
        {
            "real_tmp": np.sort(deviations),
            "simulated_tmp": np.sort(simulated_temps[lag:]),
        },
        np.linspace(0, 1, len(deviations)),
    )
    sim.plot()
    plt.show() """

    return simulated_temps


def tmpr():
    t = pd.read_excel(FILE_PATH, index_col=0)
    lag = 3
    cz = "t_1"
    y = signal.detrend(t[cz].values, type="linear")
    x = t.index
    simulations = model(x, y, cz, lag)

    return simulations


simulations = tmpr()

""" def tmpr_call(t, cz, lag):
    y = signal.detrend(t[cz].values, type="linear")
    x = t.index
    deviations, simulations = model(x, y, cz, lag)

    print(
        pd.DataFrame(
            {
                "tau": x,
                "hist_tmp": y,
                "simulated_tmp": simulations,
            },
            index=None,
        )
    ) 
    return deviations, simulations


def tmpr():
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
    lags = 3
    deviations = pd.DataFrame(index=None)
    simulations = pd.DataFrame(index=None)
    for cz in climatezones:
        dev, simtmp = tmpr_call(t, cz, lags)
        deviations = pd.concat([deviations, pd.DataFrame(dev)], axis=1)
        simulations = pd.concat([simulations, pd.DataFrame(simtmp)], axis=1)
    deviations.columns = climatezones

    for cz in climatezones:
        PCA.pca_coefficients(deviations, cz)
 """
