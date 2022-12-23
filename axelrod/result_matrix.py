import operator
import os

import pandas as pd

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
        columns_of_interest = [
            "Interaction index",
            "Player name",
            "Opponent name",
            "Score",
            "Score difference",
            "Score per turn",
            "Winner List",
        ]
        df_selected_cols = df[columns_of_interest]
        self.pd_to_file(df_selected_cols, "selected_columns_for_processing")
        new_columns = [
            "P1_vs_P2",
            "Winners[P1_P2_EQ]",
            "Utility_P1",
            "Utility_per_turn_P1",
            "Utility_P2",
            "Utility_per_turn_P2",
        ]

        df_selected_to_dict = df_selected_cols.to_dict("records")
        pair_results = {}

        for record in df_selected_to_dict:

            # print(record)

            p1 = record["Player name"]
            p2 = record["Opponent name"]
            p1_score = record["Score"]
            p1_score_per_turn = record["Score per turn"]
            score_diff = record["Score difference"]

            # set the current working pair
            pair = f"{p1},{p2}"
            inverse_pair = f"{p2},{p1}"

            if pair not in pair_results and inverse_pair not in pair_results:
                pair_results[pair] = self.init_row(pair)

            if p1 == p2:
                ### two players having the same strategy behind
                pair_results[pair] = self.identical_players(
                    pair_results[pair], p1_score, p1_score_per_turn
                )
                pair_results[pair] = self.sum_winner_lists(
                    pair_results[pair], score_diff
                )
            else:
                if pair in pair_results:
                    pair_results[pair] = self.process_p1(
                        pair_results[pair], p1_score, p1_score_per_turn
                    )
                    pair_results[pair] = self.sum_winner_lists(
                        pair_results[pair], score_diff
                    )
                if inverse_pair in pair_results:
                    pair_results[inverse_pair] = self.process_p2(
                        pair_results[inverse_pair], p1_score, p1_score_per_turn
                    )
                    # pair_results[inverse_pair] = self.sum_winner_lists(pair_results[inverse_pair], score_diff)

        post_tour_results = pd.DataFrame.from_dict(
            pair_results, orient="index", columns=new_columns
        )
        self.pd_to_file(post_tour_results, "post_tour_results_test")

    def init_row(self, pair) -> list:
        """
        Process the results from specific columns the initial result set.
        ____
        Returns:
            list: containing all the results for the
        """
        return [pair, [0, 0, 0], [], [], [], []]

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

    def process_p1(self, row, score, turn_score):
        row[2].append(score)
        row[3].append(turn_score)
        return row

    def process_p2(self, row, score, turn_score):
        row[4].append(score)
        row[5].append(turn_score)
        return row

    def sum_winner_lists(self, row, score_diff):
        """Adds winner lists based on the score diff of the interaction."""
        if sum(row[1]) < self.repetitions:
            single_list = self.compute_interaction_list(score_diff)
            row[1] = list(map(operator.add, row[1], single_list))

        return row

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
        """Write pd objects containing the averages to csv file."""
        if not os.path.exists("results_{}/".format(self.tour_type)):
            os.makedirs("results_{}/".format(self.tour_type))

        pd.to_csv(
            "results_{}/{}_{}.csv".format(self.tour_type, self.run_scope, pd_type),
            index=False,
        )
