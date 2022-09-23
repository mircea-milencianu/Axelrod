import csv
import operator
import ast
import os
import math
import itertools

from collections import Counter, namedtuple
from multiprocessing import cpu_count
from textwrap import indent
from typing import List

import pprint as p

import pandas as pd
import numpy as np
import tqdm
from axelrod import tournament

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
        run_scope,
        tour_type,
        progress_bar,
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
        self.unique_pairs = set()
        
        self.run_scope = run_scope
        self.tour_type = tour_type

        if progress_bar:
            self.progress_bar = tqdm.tqdm(total=25, desc="Analysing")
        
        df = pd.read_csv(self.filename)
        self.process_tournament_results(df)
        
        if progress_bar:
            self.progress_bar.close()

    
    def process_tournament_results(self, df):
        """
        Build the winner dataframe.

        Parameters
        ----------
            df: a pandas DataFrame holding the tournament results
        """
        ####
        # PRE-PROCESSING results and setting the data types
        ####
        columns_of_interest = ['Interaction index', 'Player name', 'Opponent name', 'Score', 'Score difference', 'Score per turn', 'Winner List']
        df_selected_cols = df[columns_of_interest]
        self.pd_to_file(df_selected_cols, "selected_columns_for_processing")
        new_columns = [
            'P1_vs_P2',
            'Winners[P1_P2_EQ]', 
            'Utility_P1',
            'Utility_per_turn_P1',
            'Utility_P2',             
            'Utility_per_turn_P2' 
            ]

        df_selected_to_dict = df_selected_cols.to_dict('records')
        pair_results = {}

        for record in df_selected_to_dict:
            
            # print(record)
            
            p1 = record['Player name']
            p2 = record['Opponent name']
            p1_score = record['Score']
            p1_score_per_turn = record['Score per turn']

            # set the current working pair
            pair = f"{p1},{p2}"
            inverse_pair = f"{p2},{p1}"
            
            if pair not in pair_results and inverse_pair not in pair_results:
                pair_results[pair] = self.init_row(pair)
            
            if p1 == p2:
                ### two players having the same strategy behind
                pair_results[pair] = self.identical_players(pair_results[pair], p1_score, p1_score_per_turn)
            else:
                ## if pair in is current and in matrix process for p1
                ## if inverse is in dict process for p2
                pair_results[pair] = self.identical_players(pair_results[pair], p1_score, p1_score_per_turn)

            # if pair in pair_results:
            #     pair_results[pair] = self.process_p1(pair_results[pair], record['Score'], record['Score per turn'])

            # else:
            #     pair_results[pair] = 
            
            # post_tour_results = post_tour_results.concat(a_series, ignore_index=True)
        
        # print(pair_results)
        post_tour_results = pd.DataFrame.from_dict(pair_results, orient='index', columns=new_columns)
        self.pd_to_file(post_tour_results, "post_tour_results_test")
               



            # print("pair: ({},{})".format(player_name, opponent_name))
            # utility_per_turn.at[player_name, opponent_name] = self.build_interaction_vector(
            #     round(row["Score per turn"], 6), utility_per_turn.at[player_name, opponent_name]
            # )
            # utility_vectors.at[player_name, opponent_name] = self.build_interaction_vector(
            #     row["Score"], utility_vectors.at[player_name, opponent_name]
            # )
            # winners.at[player_name, opponent_name] = self.sum_lists(
            #     row["Score difference"], winners.at[player_name, opponent_name]
            # )

        # self.divide_main_diagonal(winners)

        # self.pd_to_file(winners, "winners")
        # self.pd_to_file(utility_vectors, "utility_vectors")
        # self.pd_to_file(utility_per_turn, "utility_per_turn_vectors")
        # try: 
        #     self.pd_to_file(self.correct_main_diagonal(utility_vectors, "utility_vectors"),  "inv_main_diag_utility_vectors")
        # except TypeError:
        #     print("Elements in the dataframe are not subscriptable. Check traceback.")
        # self.pd_to_file(self.correct_main_diagonal(utility_per_turn, "utility_per_turn"), "inv_main_diag_utility_per_turn_vectors")


    def init_row(self, pair) -> list:
        """
        Process the results from specific columns the initial result set.
        ____
        Returns:
            list: containing all the results for the 
        """
        return [pair, [], [], [], [], []]

    def identical_players(self, row, score, turn_score):
        """
        This is a bit tricky to solve right away. So here we have a compromise solution
        row[2] : P1 score;
        row[3] : P1 score per turn;
        row[4] : P2 score;
        row[5] : P2 score per turn;

        Here the lengths of the list are compared because the name of the strategies are the same.
        Thus, when we reach a second row with the same strategies, we KNOW that 
        we are working with the values of the second player in a pair.
        """
        if len(row[2]) > len(row[4]) and len(row[3]) > len(row[5]):
            row[4].append(score)
            row[5].append(turn_score)
        else:
            row[2].append(score)
            row[3].append(turn_score)
        
        return row


    # def process_p1(self, row, score, turn_score):
    #     row[2].append(score)
    #     row[3].append(turn_score)
    #     return row

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

    def sum_winner_lists(self, scores_diff, old_results):
        """
        Creates and adds the lists based on the score diff
        of the interaction.
        """
        if old_results is np.nan:
            old_results = [0,0,0]
            single_list = self.compute_interaction_list(scores_diff)
            new_list = list(map(operator.add, old_results, single_list))
        else:
            single_list = self.compute_interaction_list(scores_diff)
            new_list = list(map(operator.add, old_results, single_list))
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
            print("player: ",p, "; value: ", df.at[p.name, p.name])
            main_diag_df.at[p.name, column_name] = df.at[p.name, p.name][1::2]
            # print(temp_series)
            # print(df.at[p.name, p.name])
            df.at[p.name, p.name] = df.at[p.name, p.name][::2] # going over a row in the main diagonal and selecting the second elements
        
        return main_diag_df

    def divide_main_diagonal(self, df):
        """divide the elements within the list of the main diagonal"""
        for p in self.players:
            df.at[p.name, p.name] = [int(x / 2) for x in df.at[p.name, p.name]]

    def init_custom_value(self, df, value):
        """Initialize """
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
        #     "results_firstAndSecond/deviation={}/{}_{}.csv".format(self.deviation, self.run_scope, pd_type)
        # )
        if not os.path.exists("results_{}/".format(self.tour_type )):
            os.makedirs("results_{}/".format(self.tour_type))

        pd.to_csv(
            "results_{}/{}_{}.csv".format(self.tour_type, self.run_scope, pd_type),
            index=False
        )
        # self.df.to_csv("results/normed_{}.csv".format( ))
