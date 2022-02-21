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
# # Animating the French deaths
#
#
# ## Introduction
#
# In this notebook, we use open data from the French National Institute of Statistics and Economic Studies ([INSEE](https://www.insee.fr/en/accueil)) in order to animate the age distribution of deaths over time for the last decades.
# We also want to compare the women and men distributions.
# We make use of [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) libraries.
#
# The dataset is downloaded from the "[Fichier des personnes décédés](https://www.data.gouv.fr/fr/datasets/fichier-des-personnes-decedees/)".
# The [agregated version](https://www.data.gouv.fr/fr/datasets/fichier-des-personnes-decedees/#resource-f5465d95-e0f3-42b4-9bed-9db2c0d40261-header) (1971-2019) is the 1.9Gb file called `insee_deces.csv`.
# This dataset is reliable on the 1971-2019 period.

# %% [markdown]
# ## Some python code to process data
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
BY_YEAR_FILE = "deaths_by_year.png"
DEATHS_ANIMATION_FILE = "deaths.gif"
WM_DEATHS_ANIMATION_FILE = "deaths_wm.gif"
BINS = 120


# %% [markdown]
# We define an utilitary function for reading large CSV files with Pandas.

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


# %% [markdown]
# The following dataclass will store women and men data.

# %%
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
# We now define a class that will handle data loading, processing and plotting.

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

    def plot_by_year(self, output_file=BY_YEAR_FILE):
        """Plot the number of deceases by year"""
        # Add an "annee_deces" column
        self.df['annee_deces'] = self.df['date_deces'].dt.year
        women = self.df[self.df.sexe == 2]
        men = self.df[self.df.sexe == 1]
        by_year = self.df.groupby('annee_deces').size()
        by_year_women = women.groupby('annee_deces').size()
        by_year_men = men.groupby('annee_deces').size()
        ax = by_year.plot(label="Total")
        by_year_women.plot(style='--', ax=ax, label="Women")
        by_year_men.plot(style='-.', ax=ax, label="Men")
        ax.set_ybound(lower=0)
        ax.set_xlabel("Year")
        ax.set_title("Number of deaths")
        ax.legend()
        fig = ax.get_figure()

        if output_file:
            fig.savefig(output_file)
            print(f"Written {output_file}")
        return fig, ax

    def plot_year(self, year: int):
        """Plot one year histogram ad return fig, ax and bar_container"""
        # Initialize figure
        fig, ax = plt.subplots()
        ax.set_xlabel('Age of death (years)')
        _, _, bar_container = ax.hist(self.get_year(self.df, year)['age'],
                                      self.bins,
                                      range=[0, self.bins])
        fig.suptitle("Distribution of the age of death in France")
        ax.set_title(f"Year of death: {year}")
        ax.set_xlim([0, self.bins])
        ax.set_ylim([0, 25000])
        ax.set_ylabel("Number of deaths")

        return fig, ax, bar_container

    def animate(self, output_file=DEATHS_ANIMATION_FILE, hlines=False,
                **kwargs) -> animation.FuncAnimation:
        """
        Create and return animation for the total number of deaths.
        Save the animation if output_file is not empty
        """

        def update(bar_container):
            """Update ax for animation"""

            def do_animate(year):
                age = self.get_year(self.df, year)['age']
                print(f"{year}: {len(age)}", end='\r')
                n, _ = np.histogram(age, self.bins, range=[0, self.bins])
                ax.set_title(f"Year of death: {year}")
                if hlines:
                    for birthyear in 1916, 1946:
                        age = year - birthyear
                        ax.vlines((age, ), 0, 25000, colors='r', linewidth=1,
                                linestyle='dotted')
                        ax.text(age - 10, 23000, f'Born in {birthyear}', color='r',
                                bbox=dict(facecolor='white', alpha=0.75))
                for count, rect in zip(n, bar_container.patches):
                    rect.set_height(count)
                return bar_container.patches

            return do_animate

        fig, ax, bar_container = self.plot_year(self.years[0])  # First plot
        # Build animation
        anim = animation.FuncAnimation(fig, update(bar_container), self.years,
                                       blit=True, repeat=False)
        if output_file:
            start = time.perf_counter()
            anim.save(output_file, **kwargs)
            end = time.perf_counter()
            print(f"Animation created in {end - start:0.1f} seconds "
                  f"and written to {output_file}")

        return anim

    def plot_year_wm(self, year: int):
        """Plot one year histogram and return fig and Datasets"""
        # Initialize figure
        fig, axes = plt.subplots(ncols=2, sharey=True)
        fig.suptitle(f"Distribution of the age of death in France: {year}")

        w = Dataset(name="Women", data=self.women, ax=axes[0])
        m = Dataset(name="Men", data=self.men, ax=axes[1])

        for s in w, m:
            _, _, s.bar_container = s.ax.hist(
                self.get_year(s.data, year)['age'],
                self.bins,
                range=[0, self.bins],
                orientation='horizontal')
            s.set_ax_params(self.bins, 15000, "Number of deaths")

        # These parameters are specific to left plot
        w.ax.invert_xaxis()
        w.ax.set_ylabel("Age of death (years)")

        fig.tight_layout()
        return fig, (w, m)

    def animate_wm(self, output_file=WM_DEATHS_ANIMATION_FILE,
                   **kwargs) -> animation.FuncAnimation:
        """
        Create and return animation for women/men plots
        Save the animation if output_file is not empty
        """

        def update(wm):
            """Update ax for animation"""

            def do_animate(year):
                fig.suptitle(
                    f"Distribution of the age of death in France: {year}")
                for s in wm:
                    age = self.get_year(s.data, year)['age']
                    s.len = len(age)
                    n, _ = np.histogram(age, self.bins, range=[0, self.bins])
                    for count, rect in zip(n, s.bar_container.patches):
                        rect.set_width(count)
                print(f"{year}: {wm[0].len} women, {wm[1].len} men", end='\r',
                      flush=True)
                return s.bar_container.patches

            return do_animate

        fig, wm = self.plot_year_wm(self.years[0])  # First plot
        # Build animation
        anim = animation.FuncAnimation(fig, update(wm), self.years,
                                       blit=True, repeat=False)
        if output_file:
            start = time.perf_counter()
            anim.save(output_file, **kwargs)
            end = time.perf_counter()
            print(f"Animation created in {end - start:0.1f} seconds "
                  f"and written to {output_file}")

        return anim


# %% [markdown]
# The following main functions are useful for a CLI usage.

# %%
def total_by_year(csv_file=CSV_FILE, bins=BINS,
                  output_file=BY_YEAR_FILE):
    """Plot total deaths by year"""
    data = Data(csv_file, bins)
    data.plot_by_year(output_file)


def animate_total(csv_file=CSV_FILE, bins=BINS,
                  output_file=DEATHS_ANIMATION_FILE):
    """Animate the total number deaths over the years"""
    data = Data(csv_file, bins)
    data.animate(output_file)


def animate_wm(csv_file=CSV_FILE, bins=BINS,
               output_file=WM_DEATHS_ANIMATION_FILE):
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
    # exit because the code below is designed for being executed
    # in a notebook
    exit() 

# %% [markdown]
# ## Processing INSEE data
#
#
# ### Data loading
#
# We now load the data from the CSV file.

# %%
data = Data()

# %% [markdown]
# ### Deaths over the years
#
# Let's plot the number of deaths over the year during the target period.

# %%
data.plot_by_year(output_file=None);

# %% [markdown]
# ### Animating the total deaths
#
# We loop over the 1971-2019 period in order to create a matplotlib animation object then we display it.

# %%
anim = data.animate(output_file=None)
video = anim.to_jshtml(default_mode='loop')
plt.close()
HTML(video)

# %% [markdown]
# ### Comparing women and men

# %%
anim_wm = data.animate_wm(output_file=None)
video_wm = anim_wm.to_jshtml(default_mode='loop')
plt.close()
HTML(video_wm)

# %% [markdown]
# We finally save the animation to a gif file.

# %%
data.animate_wm(fps=12)
plt.close()

# %%
# %matplotlib notebook
year = 1975
fig, ax, bar_container = data.plot_year(year)
for birthyear in 1916, 1946:
    age = year - birthyear
    ax.vlines((age, ), 0, 25000, colors='r', linewidth=1, linestyle='dotted')
    ax.text(age - 10, 23000, f'Born in {birthyear}', color='r', bbox=dict(facecolor='white', alpha=0.75))


# %%

# %%
plt.show()

# %%
