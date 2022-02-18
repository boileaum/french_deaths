# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visualize of French deaths data
#
# The dataset is downloaded from the "[Fichier des personnes décédés](https://www.data.gouv.fr/fr/datasets/fichier-des-personnes-decedees/)".
# The [agregated version](https://www.data.gouv.fr/fr/datasets/fichier-des-personnes-decedees/#resource-f5465d95-e0f3-42b4-9bed-9db2c0d40261-header) (1971-2019) is the 1.9Gb file called `insee_deces.csv`.
#
# Starting with some usual initializations.

# %%
"""Visualize French deaths data"""

from IPython.display import HTML
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import fire
import math
import seaborn as sns
from dataclasses import dataclass

sns.set()
plt.rcParams['figure.figsize'] = [10, 8]
CSV_FILE = "insee_deces.csv"
BINS = 120

# %% [markdown]
# On se définit une fonction utilitaire pour lire de gros fichiers CSV avec Pandas.

# %%


def read_csv_in_chunks(path, n_lines, **read_params):
    """
    A wrapper over pandas.read_csv() that prints out a progress bar
    (source: https://gist.github.com/f-huang/d2a949ecc37ec714e198c45498c0b779)
    """
    if 'chunksize' not in read_params or read_params['chunksize'] < 1:
        read_params['chunksize'] = 80000
    chunks = [0] * math.ceil(n_lines / read_params['chunksize'])
    for i, chunk in enumerate(pd.read_csv(path, **read_params)):
        percent = min(
            ((i + 1) * read_params['chunksize'] / n_lines) * 100, 100.0)
        print("#" * int(percent), f"{percent:.2f}%", end='\r', flush=True)
        chunks[i] = chunk
    df = pd.concat(chunks, axis=0)
    del chunks
    print()
    return df


@dataclass
class Dataset:
    """Class for storing a named dataset and its visual elements"""
    name: str
    data: pd.DataFrame
    ax: matplotlib.axes.SubplotBase
    barcontainer: matplotlib.container.BarContainer = None

    def set_ax_params(self, bins: int, xmax: int, xlabel: str):
        """Set parameters to ax"""
        self.ax.set_title(self.name)
        self.ax.set_ylim([0, bins])
        self.ax.set_xlabel(xlabel)
        self.ax.set_xlim([0, xmax])


# %% [markdown]
# We define a class that will handle data loading and processing.

# %%


class Data:
    """A class to handle decease data"""

    years = tuple(range(1972, 2020))

    def __init__(self, csv_file=CSV_FILE, bins=BINS):
        self.csv_file = csv_file
        self.bins = bins
        self.load_csv()

    def load_csv(self):
        """Load csv as pandas dataframes"""

        print(f"Loading {self.csv_file}")
        start = time.perf_counter()
        num_lines = sum(1 for _ in open('insee_deces.csv')) - 1
        print(f"File contains {num_lines:,} entries")
        df = read_csv_in_chunks(self.csv_file, num_lines,
                                usecols=[2, 3, 7],
                                parse_dates=['date_naissance', 'date_deces'],
                                infer_datetime_format=True,
                                date_parser=lambda x: pd.to_datetime(
                                    x, errors='coerce'),
                                na_filter=False,
                                chunksize=200000)
        end = time.perf_counter()
        print(f"File loaded in {end - start:0.1f} seconds")
        print("Processing dataframe")
        # remove entries containing missing values
        df.dropna(axis='index', inplace=True)
        df['age'] = (df['date_deces'] - df['date_naissance']) / \
            np.timedelta64(1, 'Y')
        df = df[df['age'] >= 0]
        df = df[min(self.years) <= df['date_deces'].dt.year]
        df = df[df['date_deces'].dt.year <= max(self.years)]
        self.women = df[df.sexe == 2]
        self.men = df[df.sexe == 1]
        self.df = df
        print(f"Dataframe contains {len(df):,} entries")

    @staticmethod
    def get_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Return sub dataframe corresponding to decease year"""
        return df[df['date_deces'].dt.year == year]

    def plot_by_year(self, by_year_file="deces_par_annee.png"):
        """Plot the number of deceases by year"""
        # Add an "annee_deces" column
        self.df['annee_deces'] = self.df['date_deces'].dt.year
        women = self.df[self.df.sexe == 2]
        men = self.df[self.df.sexe == 1]
        by_year = self.df.groupby('annee_deces').size()
        by_year_women = women.groupby('annee_deces').size()
        by_year_men = men.groupby('annee_deces').size()
        ax = by_year.plot(label="Total")
        by_year_women.plot(style='--', ax=ax, label="Femmes")
        by_year_men.plot(style='-.', ax=ax, label="Hommes")
        ax.set_ybound(lower=0)
        ax.set_xlabel("Année")
        ax.set_title("Nombre de décès")
        ax.legend()
        fig = ax.get_figure()

        if by_year_file:
            fig.savefig(by_year_file)
            print(f"Written {by_year_file}")
        return fig, ax

    def plot_year(self, year: int):
        """Plot one year histogram ad return fig, ax and bar_container"""
        # Initialize figure
        fig, ax = plt.subplots()
        ax.set_xlabel('Âge (années)')
        _, _, bar_container = ax.hist(self.get_year(self.df, year)['age'],
                                      self.bins,
                                      range=[0, self.bins])
        fig.suptitle("Répartition de l'âge de décès en France")
        ax.set_title(f"Année de décès : {year}")
        ax.set_xlim([0, self.bins])
        ax.set_ylim([0, 25000])
        ax.set_ylabel("Nombre de décès")

        return fig, ax, bar_container

    def animate(self, output_file="deces.gif",
                save=True) -> animation.FuncAnimation:
        """Create and return animation"""

        def update(bar_container):
            """Update ax for animation"""

            def do_animate(year):
                age = self.get_year(self.df, year)['age']
                print(f"{year}: {len(age)}\r")
                n, _ = np.histogram(age, self.bins, range=[0, self.bins])
                ax.set_title(f"Année du décès : {year}")
                ax.set_xlim([0, self.bins])

                for count, rect in zip(n, bar_container.patches):
                    rect.set_height(count)
                return bar_container.patches

            return do_animate

        fig, ax, bar_container = self.plot_year(self.years[0])  # First plot
        # Build animation
        anim = animation.FuncAnimation(fig, update(bar_container), self.years,
                                       blit=True, repeat=False)
        if save:
            start = time.perf_counter()
            anim.save(output_file)
            end = time.perf_counter()
            print(f"Animation created in {end - start:0.1f} seconds "
                  f"and written to {output_file}")

        return anim

    def plot_year_wm(self, year: int):
        """Plot one year histogram and return fig and Datasets"""
        # Initialize figure
        fig, axes = plt.subplots(ncols=2, sharey=True)
        fig.suptitle(f"Répartition de l'âge de décès en France : {year}")

        w = Dataset(name="Femmes", data=self.women, ax=axes[0])
        m = Dataset(name="Hommes", data=self.men, ax=axes[1])

        for s in w, m:
            _, _, s.bar_container = s.ax.hist(
                self.get_year(s.data, year)['age'],
                self.bins,
                range=[0, self.bins],
                orientation='horizontal')
            s.set_ax_params(self.bins, 15000, "Nombre de décès")

        # These parameters are specific to left plot
        w.ax.invert_xaxis()
        w.ax.set_ylabel("Âge de décès")

        fig.tight_layout()
        return fig, (w, m)

    def animate_wm(self, output_file="deces_hf.gif",
                   save=True) -> animation.FuncAnimation:
        """Create and return animation for women/men plots"""

        def update(wm):
            """Update ax for animation"""

            def do_animate(year):
                fig.suptitle(
                    f"Répartition de l'âge de décès en France : {year}")
                for s in wm:
                    age = self.get_year(s.data, year)['age']
                    print(f"{year} - {s.name}: {len(age)}\r")
                    n, _ = np.histogram(age, self.bins, range=[0, self.bins])
                    for count, rect in zip(n, s.bar_container.patches):
                        rect.set_width(count)
                return s.bar_container.patches

            return do_animate

        fig, wm = self.plot_year_wm(self.years[0])  # First plot
        # Build animation
        anim = animation.FuncAnimation(fig, update(wm), self.years,
                                       blit=True, repeat=False)
        if save:
            start = time.perf_counter()
            anim.save(output_file)
            end = time.perf_counter()
            print(f"Animation created in {end - start:0.1f} seconds "
                  f"and written to {output_file}")

        return anim


# %% [markdown]
# This main function is useful for a CLI usage


def total_by_year(csv_file=CSV_FILE, bins=BINS):
    """Plot total deaths by year"""
    data = Data(csv_file, bins)
    data.plot_by_year()


def animate_total(csv_file=CSV_FILE, bins=BINS,
                  output_file="deces.gif"):
    """Animate the total number deaths over the years"""
    data = Data(csv_file, bins)
    data.animate(output_file)


def animate_wm(csv_file=CSV_FILE, bins=BINS,
               output_file="deces_wm.gif"):
    """Animate the number of women and men deaths over the years"""
    data = Data(csv_file, bins)
    data.animate_wm(output_file)


# execution is in the command line
if __name__ == '__main__' and '__file__' in globals():
    fire.Fire({
        'total_by_year': total_by_year,
        'animate_total': animate_total,
        'animate_wm': animate_wm
    })
    exit()

# %%
data = Data()

# %%
# data.plot_by_year(by_year_file=None)
data.plot_by_year()

# %% [markdown]
# We create an animation object

# %%
anim = data.animate(save=False)

# %%
video = anim.to_html5_video()

# %%
HTML(video)

# %%
