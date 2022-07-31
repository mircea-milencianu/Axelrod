import csv
import operator
import ast
import os
import math

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


class ResultMatrix:
    """
    A class to build the winner matrix for a tournament.
    Reads in a CSV file produced by the tournament class.

    """

    def __init__(
        self,
        filename,
        players,
        repetitions,
        deviation,
        run_type,
        processes=None,
        progress_bar=False,
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
        self.player_names = [player.name for player in players]
        self.deviation = deviation
        self.run_type = run_type
        if progress_bar:
            self.progress_bar = tqdm.tqdm(total=25, desc="Analysing")
        # temp_df = self.df["Winner List"].apply((lambda x: ast.literal_eval(x)))
        # self.df.update(temp_df)

    # def calc_winner(self):
    #     """
    #     Calculates the winner for a player -- opponent pair.
    #     """

    #         if row["Player name"] == player and row["Opponent name"] == opponent:
    #             winner_list =list(map(operator.add, winner_list, row["Winner List"]))
    #     # p.pprint("{} against {}".format(player, opponent))
    #     # p.pprint("The final result for the interaction: {}".format(winner_list))
    #     return winner_list

    def create(self):
        """Create the matrix"""
        self.df = pd.read_csv(self.filename)
        self.build_winner_pd()


    def build_winner_pd(self):
        """
        Build the winner dataframe.

        Parameters
        ----------
            df: a pandas DataFrame holding the tournament results
        """
        # player_names = self.df["Player name"].unique()

        winners = pd.DataFrame(index=self.player_names, columns=self.player_names)
        utility_vectors = pd.DataFrame(
            index=self.player_names, columns=self.player_names
        )
        utility_per_turn = pd.DataFrame(
            index=self.player_names, columns=self.player_names
        )
        self.init_custom_value(winners, [0, 0, 0])
        # basically going over every row and applying the functions we need for creating
        # meaningful matrices; also theres a trick to keep the names the same, 
        # as some differences might appear because of different configuration parameters;
        for _, row in self.df.iterrows():
            for pl in self.player_names:
                if pl in row["Player name"]:
                    player_name = pl
                if pl in row["Opponent name"]:
                    opponent_name = pl
            utility_per_turn.at[player_name, opponent_name] = self.build_interaction_vector(
                round(row["Score per turn"], 6), utility_per_turn.at[player_name, opponent_name]
            )
            utility_vectors.at[player_name, opponent_name] = self.build_interaction_vector(
                row["Score"], utility_vectors.at[player_name, opponent_name]
            )
            winners.at[player_name, opponent_name] = self.sum_lists(
                row["Score difference"], winners.at[player_name, opponent_name]
            )

        self.divide_main_diagonal(winners)
        self.pd_to_file(self.correct_main_diagonal(utility_vectors, "utility_vectors"),  "inv_main_diag_utility_vectors")
        self.pd_to_file(self.correct_main_diagonal(utility_per_turn, "utility_per_turn"), "inv_main_diag_utility_per_turn_vectors")

        self.pd_to_file(winners, "winners")
        self.pd_to_file(utility_vectors, "utility_vectors")
        self.pd_to_file(utility_per_turn, "utility_per_turn_vectors")



    def build_interaction_vector(self, interaction_score, current_vector):
        """
        Builds a list containing all the payoffs from this pairs interaction
        """
        # print(interaction_score)
        if isinstance(current_vector, list):
            current_vector.append(interaction_score)
        elif math.isnan(current_vector):
            current_vector = []
            current_vector.append(interaction_score)
        # print(current_vector)
        return current_vector

    def sum_lists(self, scores_diff, temp_list):
        """
        Creates and adds the lists based on the score diff
        of the interaction.
        """
        single_list = self.compute_interaction_list(scores_diff)
        new_list = list(map(operator.add, temp_list, single_list))
        # p.pprint(new_list)

        return new_list

    def correct_main_diagonal(self, df, df_name):
        """
        eliminate duplicate elements from the main diagonal vectors
        slicing ranges pandas -- [start:stop:step]

        based on the order of the interactions 
        """
        column_name = "{} inverse main diagonal".format(df_name)
        main_diag_df = pd.DataFrame(index=self.player_names ,columns=[column_name])
        # print(main_diag_df)
        for p in self.players:
            main_diag_df.at[ p.name, column_name] = df.at[p.name, p.name][1::2]
            # print(temp_series)
            # print(df.at[p.name, p.name])
            df.at[p.name, p.name] = df.at[p.name, p.name][::2] # going over a row in the main diagonal and selecting the second elements
        
        return main_diag_df

    def divide_main_diagonal(self, df):
        """divide the elements within the list of the main diagonal"""
        for p in self.players:
            df.at[p.name, p.name] = [int(x / 2) for x in df.at[p.name, p.name]]

    def init_custom_value(self, df, value):
        """Add the value to each element of the matrix."""
        for p1 in self.players:
            for p2 in self.players:
                # p.pprint(p1.name)
                df.at[p1.name, p2.name] = value

    def compute_interaction_list(self, scores_diff):
        """Returns the index of the winner of the Match"""

        if scores_diff is not None:
            if scores_diff == 0:
                return [0, 0, 1]  # No winner
            if scores_diff > 0:
                return [1, 0, 0]
            elif scores_diff < 0:
                return [0, 1, 0]

    def pd_to_file(self, pd, pd_type):
        """
        Write pd objects containing the averages to csv file.
        """
        # if not os.path.exists("results_firstAndSecond/deviation={}/".format(self.deviation)):
        #     os.makedirs("results_firstAndSecond/deviation={}/".format(self.deviation))

        # pd.to_csv(
        #     "results_firstAndSecond/deviation={}/{}_{}.csv".format(self.deviation, self.run_type, pd_type)
        # )
        if not os.path.exists("results_dev/deviation={}/".format(self.deviation)):
            os.makedirs("results_dev/deviation={}/".format(self.deviation))

        pd.to_csv(
            "results_dev/deviation={}/{}_{}.csv".format(self.deviation, self.run_type, pd_type)
        )
        # self.df.to_csv("results/normed_{}.csv".format( ))
