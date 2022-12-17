import numpy as np
import networkx as nx
import pandas as pd
from operator import itemgetter
import operator
from tqdm import tqdm


def random_landmark_selection(graph, num_landmarks):
    """
    This function randomly selects landmarks and computes the distances from each landmark to every reachable node in the graph.

    :param graph (nx.Graph): the networkx graph created from the edgelist
    :param num_landmarks (int): number of landmarks selected.

    :returns:
      - distance_mapping (pandas.core.frame.DataFrame): a pandas DataFrame having as index column the list of all nodes in
                                                        the graph. Each column other than the index column is denoted by a
                                                        landmark and contains the distance from the landmark to every other node.
                                                        If a node is not reachable by a landmark, the value in the cell will
                                                        be NaN.

    """
    nodelist = np.array(list(graph.nodes()))
    # randomly select landmarks
    landmarks = np.random.choice(np.array(list(graph.nodes())), num_landmarks)

    # this variable will contain the distances between landmarks and all other nodes
    # if a node is not reachable, the distance will be set to NaN
    distance_mapping = pd.DataFrame()
    distance_mapping["vertices"] = list(graph.nodes())
    distance_mapping = distance_mapping.set_index("vertices")

    for u in landmarks:
        # get the distance from "u" to all other nodes
        shortest_path_lengths = nx.single_source_shortest_path_length(graph, u)
        sp_array = np.array(list(shortest_path_lengths.items()))

        # get an array of reached nodes
        reached_nodes = list(shortest_path_lengths.keys())
        recorded_distances = list(shortest_path_lengths.values())

        df_u = pd.DataFrame()
        df_u["vertices"] = reached_nodes
        df_u[str(u)] = recorded_distances

        distance_mapping = distance_mapping.join(df_u.set_index("vertices"))


    return distance_mapping


def run_baseline_strategy(paths, save_directory_baseline):
    """
    This function is used to perform the pre-set experiments using the baseline strategy (i.e. random landmark
    selection). This function integrates certain common functions designed initially for the Recursive Node Elimination
    strategy (i.e. strategy1.py).

    :param paths: dictionary containing, as keys, the name of the graphs and, as values, the paths towards the edgelists
                  in .csv format

    :param save_directory_baseline: string denoting the path to the directory in which the results are saved
    """
    seeds = [50, 51, 52, 53, 54]
    n_landmarks = [40, 60, 80, 100, 120]

    # run experiments:
    for graph_key in paths.keys():
        # create variable in which to save the results
        results_df = pd.DataFrame()
        results_df["number_of_landmarks"] = n_landmarks

        print(f"Currently working on graph: {graph_key} ....")
        df = pd.read_csv(paths[graph_key]).loc[:, ["source", "target"]]
        graph = nx.from_pandas_edgelist(df, "source", "target", create_using=nx.Graph)
        print("created graph...")

        # make 2 variables for saving the average approximation error and the average L/U ratio
        avg_approx_error = []
        avg_L_U = []

        for num_l in n_landmarks:
            print(f"Currently selecting {num_l} landmarks")

            tmp_approx = []
            tmp_LU = []

            for s in tqdm(seeds):
                np.random.seed(s)
                # obtain results for strategy 1: recursive node elimination
                distance_mapping = random_landmark_selection(graph, num_l)

                mean_approximation_error, mean_L_U = compute_metrics_strategy1(graph, distance_mapping, N=500)

                tmp_approx.append(mean_approximation_error)
                tmp_LU.append(mean_L_U)

            avg_approx_error.append(np.mean(np.array(tmp_approx)))
            avg_L_U.append(np.mean(np.array(tmp_LU)))
            print(f"avg approx error: {np.mean(np.array(tmp_approx))}")

        results_df["approximation_error"] = avg_approx_error
        results_df["LU_ratio"] = avg_L_U

        save_path = save_directory_baseline + str(graph_key) + "_overall_results.csv"
        results_df.to_csv(save_path)
