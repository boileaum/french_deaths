# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.9.10 (''.venv'': venv)'
#     language: python
#     name: python3
# ---

# %%
from distutils.log import error
from pydoc import describe
from matplotlib import use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import fire
import math

plt.rcParams['figure.figsize'] = [10, 8]


def read_csv_in_chunks(path, n_lines, **read_params):
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


class Data:
    """A class to handle decease data"""

    years = tuple(range(1972, 2020))

    def __init__(self, csv_file="insee_deces.csv",
                 density=False, bins=120, fill=True, use_seaborn=True):
        self.csv_file = csv_file
        self.density = density
        self.bins = bins
        self.fill = fill

        if use_seaborn:
            import seaborn as sns
            sns.set()

        self.df = self.load_csv()

    def load_csv(self):
        """Load csv as pandas dataframe"""

        print(f"Loading {self.csv_file}")
        start = time.perf_counter()
        num_lines = sum(1 for _ in open('insee_deces.csv'))
        df = read_csv_in_chunks(self.csv_file, num_lines,
                                usecols=[2, 3, 7],
                                parse_dates=['date_naissance', 'date_deces'],
                                infer_datetime_format=True,
                                date_parser=lambda x: pd.to_datetime(
                                    x, errors='coerce'),
                                na_filter=False,
                                chunksize=200000)
        # remove entries containing missing values
        df.dropna(axis='index', inplace=True)
        df['age'] = (df['date_deces'] - df['date_naissance']) / \
            np.timedelta64(1, 'Y')
        df = df[df['age'] >= 0]
        df = df[min(self.years) <= df['date_deces'].dt.year]
        df = df[df['date_deces'].dt.year <= max(self.years)]
        end = time.perf_counter()
        print(f"File containing {len(df)} entries loaded in "
              f"{end - start:0.1f} seconds")
        return df

    def plot_hist(self, year: int):
        df_year = self.df[self.df['date_deces'].dt.year == year]
        ax = df_year['age'].hist(bins=120, fill=False, density=True)
        ax.set_xlabel('Âge (années)')
        ax.set_title(year)
        return ax

    def get_year(self, year: int) -> pd.DataFrame:
        """Return sub dataframe corresponding to decease year"""
        return self.df[self.df['date_deces'].dt.year == year]

    def plot_by_year(self, by_year_file="deces_par_annee.png"):
        """Plot the number of deceases by year"""
        # Add an "annee_deces" column
        self.df['annee_deces'] = self.df['date_deces'].dt.year
        by_year = self.df.groupby('annee_deces').size()
        ax = by_year.plot()
        ax.set_ybound(lower=0)
        ax.set_xlabel("Année")

        ax.set_title("Nombre de décès")

        fig = ax.get_figure()
        fig.savefig(by_year_file)
        print(f"Written {by_year_file}")

    def plot_year(self, year: int):
        """Plot one year histogram ad return ax and bar_container"""
        # Initialize figure
        fig, ax = plt.subplots()
        ax.set_xlabel('Âge (années)')
        _, _, bar_container = ax.hist(self.get_year(year)['age'],
                                      self.bins,
                                      range=[0, self.bins],
                                      fill=self.fill,
                                      density=self.density)
        fig.suptitle("Répartition de l'âge du décès en France")
        ax.set_title(f"Année du décès : {year}")
        ax.set_xlim([0, self.bins])

        if self.density:
            ax.set_ylabel("Densité")
        else:
            ax.set_ylabel("Nombre de décès")
            ax.set_ybound(lower=0, upper=25000)

        return fig, ax, bar_container

    def animate(self, output_file="deces.gif",
                save=True) -> animation.FuncAnimation:
        """Create and return animation"""

        def update(bar_container):
            """Update ax for animation"""

            def do_animate(year):
                age = self.get_year(year)['age']
                print(f"{year}: {len(age)}")
                n, _ = np.histogram(age, self.bins, range=[0, self.bins],
                                    density=self.density)
                ax.set_title(f"Année du décès : {year}")
                ax.set_xlim([0, self.bins])

                for count, rect in zip(n, bar_container.patches):
                    rect.set_height(count)
                return bar_container.patches

            return do_animate

        fig, ax, bar_container = self.plot_year(self.years[0])  # First plot
        # Build animation
        anim = animation.FuncAnimation(fig, update(bar_container), self.years,
                                       blit=True,
                                       repeat=False)
        if save:
            start = time.perf_counter()
            anim.save(output_file)
            end = time.perf_counter()
            print(f"Animation created in {end - start:0.1f} seconds "
                  f"and written to {output_file}")

        return anim


def main(csv_file="insee_deces.csv",
         density=False, bins=120, fill=True, use_seaborn=True,
         output_file="deces.gif"):
    data = Data(csv_file, density, bins, fill, use_seaborn)
    data.plot_by_year()
    data.animate(output_file)


if __name__ == '__main__' and '__file__' in globals():  # execution is in the command line
    fire.Fire(main)
