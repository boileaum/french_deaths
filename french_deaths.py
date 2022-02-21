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
# Starting with some usual initializations and useful parameters

# %%
"""Visualize French deaths data"""

from IPython.display import HTML
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import math
import seaborn as sns
from dataclasses import dataclass

# %matplotlib inline
# %config InlineBackend.figure_format='retina'
plt.rcParams['figure.figsize'] = [12, 8]
sns.set()

BINS = 120  # Number of age bins for histograms
YEARS = tuple(range(1972, 2020))  # the 1972-2019 period will be covered
ANNOTATIONS = {1914: '1914-1918: Word war I',
               1918: '',
               1946: '1946: Babyboom'}


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
# We now define a function to load data from a CSV file.

# %%
def load_csv(csv_file) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load csv as pandas dataframes"""

    print(f"Loading {csv_file}")
    start = time.perf_counter()
    num_lines = sum(1 for _ in open('insee_deces.csv')) - 1
    print(f"File contains {num_lines:,} entries")
    df = read_csv_in_chunks(csv_file, num_lines,
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
    df = df[min(YEARS) <= df['date_deces'].dt.year]
    df = df[df['date_deces'].dt.year <= max(YEARS)]
    # We will need to group by year of death
    df['annee_deces'] = df['date_deces'].dt.year
    women = df[df.sexe == 2]  # a subset containing women
    men = df[df.sexe == 1]  # a subste containing men
    return df, women, men


# %% [markdown]
# ## Processing INSEE data
#
#
# ### Data loading
#
# We now load the data from the CSV file.
# %%
df, men, women = load_csv("insee_deces.csv")
print(f"df:\t{len(df):,} entries")
print(f"women:\t{len(women):,} entries")
print(f"men:\t{len(men):,} entries")

# %% [markdown]
# ### Deaths over the years
#
# Let's plot the number of deaths over the year during the target period.


# %%
def get_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return sub dataframe corresponding to decease year"""
    return df[df['date_deces'].dt.year == year]


# %%
def plot_by_year(df: pd.DataFrame, women: pd.DataFrame, men: pd.DataFrame):
    """Plot the number of deceases by year"""
    by_year = df.groupby('annee_deces').size()
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
    return fig, ax


# %%
plot_by_year(df, men, women)


# %% [markdown]
# ### Animating the total deaths
#
# First write a function that creates a figure from the dataframe and a year.
# %%
def plot_year(df: pd.DataFrame, year: int, annotate=False):
    """Plot one year histogram ad return fig, ax and bar_container"""
    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_xlabel('Age of death (years)')
    _, _, bar_container = ax.hist(get_year(df, year)['age'],
                                  BINS,
                                  range=[0, BINS])
    fig.suptitle("Distribution of the age of death in France")
    ax.set_title(f"Year of death: {year}")
    ax.set_xlim([0, BINS])
    ax.set_ylim([0, 25000])
    ax.set_ylabel("Number of deaths")
    annotations = {}
    if annotate:
        for birthyear, text in ANNOTATIONS.items():
            age = year - birthyear
            vl = ax.axvline((age - 1, ), color='r', linewidth=0.75,)
                            #linestyle='dotted')
            text = ax.text(age - 10, 23000, text, color='r',
                           bbox=dict(facecolor='white', alpha=0.75))
            annotations[birthyear] = (vl, text)
    return fig, ax, (bar_container, annotations)


# %% [markdown]
# Let's try this function for the year 1983.

# %%
plot_year(df, 1983);

# %%
plot_year(df, 1983, annotate=True);


# %%
def animate(df, annotate=False) -> animation.FuncAnimation:
    """
    Create and return animation for the total number of deaths.
    """

    def update(bar_container, annotations):
        """Update ax for animation"""

        def do_animate(year):
            age = get_year(df, year)['age']
            print(f"{year}: {len(age)}", end='\r')
            n, _ = np.histogram(age, BINS, range=[0, BINS])
            ax.set_title(f"Year of death: {year}")
            if annotations:
                for birthyear, (vl, text) in annotations.items():
                    age = year - birthyear
                    vl.set_xdata(age - 1)
                    text.set_x(age - 10)
            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)
            return bar_container.patches

        return do_animate

    fig, ax, artists = plot_year(df, YEARS[0], annotate)  # First plot
    # Build animation
    anim = animation.FuncAnimation(fig, update(*artists), YEARS,
                                   blit=True, repeat=False)
    return anim


# %% [markdown]
# We loop over the 1971-2019 period in order to create a matplotlib animation object then we display it.
# %%
anim_vline = animate(df, annotate=True)
video = anim_vline.to_jshtml(default_mode='loop')
plt.close()
HTML(video)

# %% [markdown]
# ### Comparing women and men
# %% [markdown]
# The following dataclass will store women and men data.

# %%
import dataclasses

@dataclass
class Dataset:
    """Class for storing a named dataset and its visual elements"""
    name: str
    data: pd.DataFrame
    ax: matplotlib.axes.SubplotBase
    barcontainer: matplotlib.container.BarContainer = None
    annotate: str = None
    annotations: dict = dataclasses.field(default_factory=dict)


    def set_ax_params(self, bins: int, xmax: int, xlabel: str, year: int):
        """Set parameters to ax"""
        self.ax.set_title(self.name)
        self.ax.set_ylim([0, bins])
        self.ax.set_xlabel(xlabel)
        self.ax.set_xlim([0, xmax])
        if self.annotate:
            for birthyear, text in ANNOTATIONS.items():
                age = year - birthyear
                vl = self.ax.axhline((age - 1, ), color='r', linewidth=0.75,)
                                #linestyle='dotted')
                if self.annotate != 'hlines':
                    text = self.ax.text(14000, age + 1, text, color='r')
                else:
                    text = None
                self.annotations[birthyear] = (vl, text)        


# %%
def plot_year_wm(women, men, year: int, annotate=False):
    """Plot one year histogram and return fig and Datasets"""
    # Initialize figure
    fig, axes = plt.subplots(ncols=2, sharey=True)
    fig.suptitle(f"Distribution of the age of death in France in {year}")

    w = Dataset(name="Women", data=women, ax=axes[0], annotate='both')
    m = Dataset(name="Men", data=men, ax=axes[1], annotate='hlines')

    for s in w, m:
        _, _, s.bar_container = s.ax.hist(
            get_year(s.data, year)['age'],
            BINS,
            range=[0, BINS],
            orientation='horizontal')
        s.set_ax_params(BINS, 15000, "Number of deaths", year)

    # These parameters are specific to left plot
    w.ax.invert_xaxis()
    w.ax.set_ylabel("Age of death (years)")

    fig.tight_layout()
    return fig, (w, m)


# %%
fig, (w, m) = plot_year_wm(women, men, 2003, annotate=True);


# %%
def animate_wm(women: pd.DataFrame,
               men: pd.DataFrame,
               annotate=False) -> animation.FuncAnimation:
    """
    Create and return animation for women/men plots
    """

    def update(wm):
        """Update ax for animation"""

        def do_animate(year):
            fig.suptitle(
                f"Distribution of the age of death in France in {year}")
            for s in wm:
                age = get_year(s.data, year)['age']
                s.len = len(age)
                n, _ = np.histogram(age, BINS, range=[0, BINS])
                for count, rect in zip(n, s.bar_container.patches):
                    rect.set_width(count)
                if s.annotations:
                    for birthyear, (vl, text) in s.annotations.items():
                        age = year - birthyear
                        vl.set_ydata(age - 1)
                        if text:
                            text.set_y(age + 1)
            print(f"{year}: {wm[0].len} women, {wm[1].len} men", end='\r',
                  flush=True)
            return s.bar_container.patches

        return do_animate

    fig, wm = plot_year_wm(women, men, YEARS[0], annotate)  # First plot
    # Build animation
    anim = animation.FuncAnimation(fig, update(wm), YEARS,
                                   blit=True, repeat=False)
    return anim


# %%
anim_wm = animate_wm(women, men)
video_wm = anim_wm.to_jshtml(default_mode='loop')
plt.close()
HTML(video_wm)

# %% [markdown]
# We finally save the animation to a gif file.

# %%
anim_wm.save("deaths_wm.gif", fps=12)
plt.close()
