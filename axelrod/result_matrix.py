
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
        self.player_names = [player.name for player in players]
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
        winners = self.build_winner_pd()
        self.pd_to_file(winners)

    def build_winner_pd(self):
        """
        Build the winner dataframe.

        Parameters
        ----------
            df: a pandas DataFrame holding the tournament results
        """
        winners = pd.DataFrame(index=self.player_names, columns=self.player_names)
        self.init_custom_value(winners, [0,0,0])
        for _, row in self.df.iterrows():
            # p.pprint("calculating for: {} against {}".format(row["Player name"], row["Opponent name"]))
            winners.at[row["Player name"], row["Opponent name"]] = self.sum_list(
                row["Score difference"],
                winners.at[row["Player name"], row["Opponent name"]])

        self.divide_main_diagonal(winners)
        return winners

    def sum_list(self, scores_diff, temp_list):
        """ 
            Creates and adds the lists based on the score diff 
            of the interaction.
        """
        single_list = self.compute_interaction_list(scores_diff)
        new_list = list(map(operator.add, temp_list, single_list))
        # p.pprint(new_list)

        return new_list
    
    def divide_main_diagonal(self, pd):
        """divide the elements within the list of the main diagonal"""
        for p in self.players:
            
            pd.at[p.name, p.name] = [int(x/2) for x in pd.at[p.name,p.name]]

    def init_custom_value(self, pd, value):
        """Add the value to each element of the matrix."""
        for p1 in self.players:
            for p2 in self.players:
                #p.pprint(p1.name)
                pd.at[p1.name,p2.name] = value

    def compute_interaction_list(self, scores_diff):
        """Returns the index of the winner of the Match"""

        if scores_diff is not None:
            if scores_diff == 0:
                return [0,0,1]  # No winner
            if scores_diff > 0:
                return [1,0,0]
            elif scores_diff < 0:
                return [0,1,0]
  

    def pd_to_file(self, pd):
        """
        Write pd objects containing the averages to csv file.
        """
        if not os.path.exists('results/'):
            os.makedirs('results/')

        pd.to_csv("results/winners_montecarlo_{}.csv".format(self.repetitions))
        #self.df.to_csv("results/normed_{}.csv".format( ))