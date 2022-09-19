import csv
import operator
import ast
import os
import math
import itertools

from collections import Counter, namedtuple
from multiprocessing import cpu_count
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
        columns_of_interest = ['Player name', 'Opponent name', 'Score', 'Score difference', 'Score per turn']
        df_of_interest = df[columns_of_interest]

        new_columns = [
            'Pair(P1, P2)',
            'Occurence', 
            'Winners[P1, P2, EQ]', 
            'Utility for P1[vector]',  
            'Utility for P2[vector]', 
            'Utility per turn for P1[vector]',
            'Utility per turn for P2[vector]' 
            ]
        post_tour_results =  pd.DataFrame(columns=new_columns)
        
        # players = df["Player name"].unique()
        # all_pairs = self.cartesian_product_of_players(players)
        # post_tour_results["Pair(P1, P2)"] = all_pairs
        
        new_columns.remove("Pair(P1, P2)")
        for col in new_columns:
            if col == "Occurence":
                post_tour_results[col] = 0
            else:
                post_tour_results[col] = [ [] for _ in range(len(all_pairs)) ]
 
        df_dict_of_interest = df_of_interest.to_dict('records')
        for row in df_dict_of_interest:
            # print(record, type(record))
            pair = "{},{}".format(row['Player name'], row['Opponent name'])
            winners = self.sum_lists(
                row["Score difference"], winners.at[player_name, opponent_name]
            )
        # for _, row in df.iterrows():
        #     # print(row, type(row))
        #     pair_in_row = self.check_pair(row['Player name'], row['Opponent name'])
        #     for _, row in post_tour_results.iterrows():
        #         if row_pair == row["Pair(P1, P2)"]:
        #             print("row_pair: {} is equal with {}".format(row_pair, row["Pair(P1, P2)"]))
        #             # different_pairs_count += 1
        #             post_tour_results["Occurence"] += 1
        

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


    def cartesian_product_of_players(self, elements: list[str]) -> list[tuple[str, str]]:
        """ 
        Precondition: `elements` does not contain duplicates.
            Postcondition: Returns unique combinations of length 2 from `elements`.

            >>> unique_combinations(["apple", "orange", "banana"])
            [("apple", "orange"), ("apple", "banana"), ("orange", "banana")]
        """
        
        return list(itertools.combinations_with_replacement(elements, 2))


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
            "results_{}/{}_{}.csv".format(self.tour_type, self.run_scope, pd_type)
        )
        # self.df.to_csv("results/normed_{}.csv".format( ))
