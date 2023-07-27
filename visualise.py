import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
from pathlib import Path
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta, timezone
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as stats

# import seaborn as sns

ANALYSISDATAFOLDER = Path(__file__).parent / "Whitenoise Analysis"
IMAGESFOLDER = Path(__file__).parent / "Temperature Graphs"


def plot_long_graph(climate_zone: str, models: pd.Series, t: pd.DataFrame):
    colors = {"hist": "gray", "curve_fitting": "green", "pca": "#FA9610"}

    title = f"{climate_zone}-{t}"
    fig, ax = plt.subplots(figsize=(300, 10))
    fig.suptitle(title)  # , fontsize=18, y=0.95)
    for name, s in models.items():
        s.plot(ax=ax, c=colors.get(name, "gray"), label=name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGESFOLDER, f"{title}.png"))


def single_monthly_plot(
    x: pd.Series, y: pd.Series, t: pd.DataFrame, year: int, title: str
):
    x = t.copy()
    x = t.index * pd.Timedelta(days=365.25) + pd.Timestamp("2000-01-01")
    x = (x + pd.Timedelta(hours=12)).floor("D")
    x.freq = "D"
    t.index = x
    # qqplot(t[1:], line="s")
    stats.probplot(t[1:], dist="norm", plot=plt)
    plt.show()
    """
    Since t was normally distributed, its quantiles followed the quantiles of
    the theoretical distribution, so that the dots of the variable values fell
    along the standard-degree line.  """
    days = -1 * 365 * year
    plt.title("epsilon mean")
    plt.plot(x[days:], t[days:].rolling(30).mean(), "r")  # mean plot
    plt.show()
    plt.title("epsilon standard deviation")
    plt.plot(x[days:], t[days:].rolling(30).std())  # standard deviation plot
    plt.show()


def monthly_plot(t: pd.DataFrame, title: str):
    x = t.copy()
    x = t.index * pd.Timedelta(days=365.25) + pd.Timestamp("2000-01-01")
    x = (x + pd.Timedelta(hours=12)).floor("D")
    x.freq = "D"
    t.index = x

    tavg = t.groupby(lambda ts: (ts.year, ts.month)).mean()
    tavg.index = pd.MultiIndex.from_tuples(tavg.index, names=("year", "month"))
    tavg.index = tavg.index.set_names(("year", "month"))
    fig, axes = plt.subplots(1, 1, sharey=False, figsize=(15, 10))
    fig.suptitle("")
    for y, subdf in tavg.groupby("year"):
        subdf.plot(ax=axes)


def yearly_plot(tmpr: pd.DataFrame, title: str):
    x = tmpr.index * pd.Timedelta(days=365.25) + pd.Timestamp("2000-01-01")
    x = (x + pd.Timedelta(hours=12)).floor("D")
    x.freq = "D"
    tmpr.index = x
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(title, fontsize=18, y=0.95)
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

    df = tmpr.resample(rule="AS").mean()  # resampling the data into "Yearly" format

    fig.text(0.5, 0.09, "year", ha="center")
    fig.text(0.09, 0.5, "temp", va="center", rotation="vertical")

    for cz, ax in zip(climatezones, axes.ravel()):  # loop through climatezones and axes
        df[cz].hist(
            ax=ax, color="blue", linewidth=1
        )  # filter df for cz and plot on specified axes
        ax.set_title(cz)  # chart formatting
        ax.set_xlabel("")

    axes[3, 3].axis(
        "off"
    )  # since we have only 15 climatezones, so we ignore the 16th graph

    for i in range(
        4
    ):  # set all labels of the 1st row at the top and make bottom labels invisible
        axes[0, i].xaxis.set_tick_params(labeltop=True)
        axes[0, i].xaxis.set_tick_params(labelbottom=False)
        # axes[i, 0].set_ylabel("temp")

    for i in range(1, 4):
        for j in range(4):
            axes[i, j].axes.get_xaxis().set_visible(False)
    plt.show()
    # fig.savefig(os.path.join(ANALYSISDATAFOLDER, f"{title}.png"))
