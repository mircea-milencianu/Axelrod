
import csv
import itertools
import warnings
from collections import Counter, namedtuple
from multiprocessing import cpu_count
from typing import List

import pprint as p

import pandas as pd
import numpy as np
import tqdm
from axelrod.action import Action

from . import eigen

C, D = Action.C, Action.D


def update_progress_bar(method):
    """A decorator to update a progress bar if it exists"""

    def wrapper(*args, **kwargs):
        """Run the method and update the progress bar if it exists"""
        output = method(*args, **kwargs)

        try:
            args[0].progress_bar.update(1)
        except AttributeError:
            pass

        return output

    return wrapper


class WinnerMatrix:
    """
    A class to build the winner matrix for a tournament. 
    Reads in a CSV file produced by the tournament class.

    """

    def __init__(
        self, filename, players, repetitions, processes=None, progress_bar=False
    ):
        """
        Parameters
        ----------
            filename : string
                the file from which to read the interactions
            players : list
                A list of the names of players. If not known will be efficiently
                read from file.
            repetitions : int
                The number of repetitions of each match. If not know will be
                efficiently read from file.
            processes : integer
                The number of processes to be used for parallel processing
            progress_bar: boolean
                If a progress bar will be shown.
        """
        self.filename = filename
        self.players, self.repetitions = players, repetitions
        self.num_players = len(self.players)

        if progress_bar:
            self.progress_bar = tqdm.tqdm(total=25, desc="Analysing")

        df = pd.read_csv(filename)
        
        p.pprint(df.tail())
        p.pprint(df.dtypes)

        self.build_match_results_matrix(df)
        
        self.build_winner_pd(df)

        if progress_bar:
            self.progress_bar.close()

    
    def build_match_results_matrix(self, df):
        """
        Build the resutlts matrix matrix.

        Parameters
        ----------
            df: a pandas DataFrame holding the tournament results
        """
        df_by_repetions = df.groupby(["Player name"])["Repetition"].sum()
        p.pprint(df_by_repetions)

        winners = pd.DataFrame(index=self.players, columns=self.players)
        p.pprint(winners)



    def build_winner_pd(self, df):
        """
        Build the winner dataframe.

        Parameters
        ----------
            df: a pandas DataFrame holding the tournament results
        """
        df_by_players = df.groupby(["Player name","Opponent name"])["Win"].sum()
        p.pprint(df_by_players)


    def __eq__(self, other):
        """
        Check equality of results set

        Parameters
        ----------
            other : axelrod.ResultSet
                Another results set against which to check equality
        """

        def list_equal_with_nans(v1: List[float], v2: List[float]) -> bool:
            """Matches lists, accounting for NaNs."""
            if len(v1) != len(v2):
                return False
            for i1, i2 in zip(v1, v2):
                if np.isnan(i1) and np.isnan(i2):
                    continue
                if i1 != i2:
                    return False
            return True

        return all(
            [
                self.wins == other.wins,
                self.match_lengths == other.match_lengths,
                self.scores == other.scores,
                self.normalised_scores == other.normalised_scores,
                self.ranking == other.ranking,
                self.ranked_names == other.ranked_names,
                self.payoffs == other.payoffs,
                self.payoff_matrix == other.payoff_matrix,
                self.payoff_stddevs == other.payoff_stddevs,
                self.score_diffs == other.score_diffs,
                self.payoff_diffs_means == other.payoff_diffs_means,
                self.cooperation == other.cooperation,
                self.normalised_cooperation == other.normalised_cooperation,
                self.vengeful_cooperation == other.vengeful_cooperation,
                self.cooperating_rating == other.cooperating_rating,
                self.good_partner_matrix == other.good_partner_matrix,
                self.good_partner_rating == other.good_partner_rating,
                list_equal_with_nans(self.eigenmoses_rating, other.eigenmoses_rating),
                list_equal_with_nans(self.eigenjesus_rating, other.eigenjesus_rating),
            ]
        )

    def __ne__(self, other):
        """
        Check inequality of results set

        Parameters
        ----------
            other : axelrod.ResultSet
                Another results set against which to check inequality
        """
        return not self.__eq__(other)

    def summarise(self):
        """
        Obtain summary of performance of each strategy:
        ordered by rank, including median normalised score and cooperation
        rating.

        Output
        ------
            A list of the form:

            [[player name, median score, cooperation_rating],...]

        """

        median_scores = map(np.nanmedian, self.normalised_scores)
        median_wins = map(np.nanmedian, self.wins)

        self.player = namedtuple(
            "Player",
            [
                "Rank",
                "Name",
                "Median_score",
                "Cooperation_rating",
                "Wins",
                "Initial_C_rate",
                "CC_rate",
                "CD_rate",
                "DC_rate",
                "DD_rate",
                "CC_to_C_rate",
                "CD_to_C_rate",
                "DC_to_C_rate",
                "DD_to_C_rate",
            ],
        )

        states = [(C, C), (C, D), (D, C), (D, D)]
        state_prob = []
        for i, player in enumerate(self.normalised_state_distribution):
            counts = []
            for state in states:
                p = sum([opp[state] for j, opp in enumerate(player) if i != j])
                counts.append(p)
            try:
                counts = [c / sum(counts) for c in counts]
            except ZeroDivisionError:
                counts = [0 for c in counts]
            state_prob.append(counts)

        state_to_C_prob = []
        for player in self.normalised_state_to_action_distribution:
            rates = []
            for state in states:
                counts = [
                    counter[(state, C)] for counter in player if counter[(state, C)] > 0
                ]

                if len(counts) > 0:
                    rate = np.mean(counts)
                else:
                    rate = 0

                rates.append(rate)
            state_to_C_prob.append(rates)

        summary_measures = list(
            zip(
                self.players,
                median_scores,
                self.cooperating_rating,
                median_wins,
                self.initial_cooperation_rate,
            )
        )

        summary_data = []
        for rank, i in enumerate(self.ranking):
            data = list(summary_measures[i]) + state_prob[i] + state_to_C_prob[i]
            summary_data.append(self.player(rank, *data))

        return summary_data

    def write_summary(self, filename):
        """
        Write a csv file containing summary data of the results of the form:

            "Rank", "Name", "Median-score-per-turn", "Cooperation-rating", "Initial_C_Rate", "Wins", "CC-Rate", "CD-Rate", "DC-Rate", "DD-rate","CC-to-C-Rate", "CD-to-C-Rate", "DC-to-C-Rate", "DD-to-C-rate"


        Parameters
        ----------
            filename : a filepath to which to write the data
        """
        summary_data = self.summarise()
        with open(filename, "w") as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")
            writer.writerow(self.player._fields)
            for player in summary_data:
                writer.writerow(player)


def create_counter_dict(df, player_index, opponent_index, key_map):
    """
    Create a Counter object mapping states (corresponding to columns of df) for
    players given by player_index, opponent_index. Renaming the variables with
    `key_map`. Used by `ResultSet._reshape_out`

    Parameters
    ----------
        df : a multiindex pandas df
        player_index: int
        opponent_index: int
        key_map : a dict
            maps cols of df to strings

    Returns
    -------
        A counter dictionary
    """
    counter = Counter()
    if player_index != opponent_index:
        if (player_index, opponent_index) in df.index:
            for key, value in df.loc[player_index, opponent_index].items():
                if value > 0:
                    counter[key_map[key]] = value
    return counter
