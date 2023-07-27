import pandas as pd
import numpy as np
import os
from scipy import signal
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca_coefficients(deviations: pd.DataFrame, climate_zone: pd.Series) -> pd.Series:
    """This function finds the principle component analysis between two climatezones and
    stores the pca coefficients in an excel file. The historic deviations/ offtakes are
    fit using the pca coefficients and the fitted curve is returned.

    Args:
        deviations (pd.DataFrame): historic deviations
        climate_zone (pd.Series): set of two of more climatezones

    Returns:
        pd.Series: the timseries after fitting the PCA is returned
    """

    # You must normalize the data before applying the fit method
    df_normalized = (deviations - deviations.mean()) / deviations.std()

    pca = PCA(n_components=0.99).fit(
        df_normalized
    )  # PCA will select the number of components such that the amount of
    # variance that needs to be explained is greater than the percentage
    # specified by n_components

    pca_coeffs = pd.DataFrame(
        pca.components_.T,
        columns=["PC%s" % _ for _ in range(1, pca.n_components_ + 1)],
        index=deviations.columns,
    )  # Principal components correlation coefficients

    pca_coeffs.to_excel(
        "pca_coefficients.xlsx", sheet_name="pca values"
    )  # save pca coefficients in excel

    """ most_important_names = [
        df.columns[
            [np.abs(pca.components_[i]).argmax() for i in range(pca.n_components_)][i]
        ]
        for i in range(pca.n_components_)
    ]  # get the most important feature names """

    pca_components = pd.DataFrame(
        pca.transform(df_normalized),
        columns=["pc_%i" % i for i in range(1, pca.n_components_ + 1)],
        index=deviations.index,
    )  # Fit and transform data

    # plot_pca_explainedvariance(pca)

    return pca_components


def plot_pca_explainedvariance(pca):
    """From this plot, we can see that over 95% of the variance is captured
    withing the first five largest principal components and all 100% is captured
    in 11 principal components. Therefore, it is acceptable to choose the first
    11 largest components of PCA

    Using (PCA(n_components=0.99).fit(df_normalized)), the algorithm can choose
    on it's own how many number of coefficients are best represented by PCA.

    Args:
        pca: the PCA vector after fitting with n_components
    """

    title = "PCA_explained_variance"
    # plt.plot(pca.explained_variance_ratio_)
    plt.plot(
        range(1, len(pca.explained_variance_) + 1),
        np.cumsum(pca.explained_variance_),
        c="red",
        label="Cumulative Explained Variance",
    )
    plt.legend(loc="upper left")
    plt.ylabel("Explained Variance")
    plt.xlabel("Components")
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.show()
