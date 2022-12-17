import numpy as np
import networkx as nx
import pandas as pd
from operator import itemgetter
import operator
from tqdm import tqdm


def recursive_node_elimination(graph, num_landmarks = [50, 30, 15, 5]):
    """
    This function selects landmarks and computes the distances from each landmark to every reachable node in the graph.

    :param graph (nx.Graph): the networkx graph created from the edgelist
    :param num_landmarks (list): number of landmarks selected at each iteration.

    :returns:
      - distance_mapping (pandas.core.frame.DataFrame): a pandas DataFrame having as index column the list of all nodes in
                                                        the graph. Each column other than the index column is denoted by a
                                                        landmark and contains the distance from the landmark to every other node.
                                                        If a node is not reachable by a landmark, the value in the cell will
                                                        be NaN.

    """
    # initialize set of landmarks
    landmark_nodes = []

    # rank the nodes based on degree centrality
    ranking = dict(sorted(nx.degree_centrality(graph).items(), key=operator.itemgetter(1),reverse=True))

    # create a list of nodes ordered by their degree centrality
    ranking_list = list(ranking)

    current_n_landmarks = 0

    # this variable will contain the distances between landmarks and all other nodes
    # if a node is not reachable, the distance will be set to NaN
    distance_mapping = pd.DataFrame()
    distance_mapping["vertices"] = list(graph.nodes())
    distance_mapping = distance_mapping.set_index("vertices")

    for num_l in num_landmarks:
        # get the top num_l landmarks
        u_list = ranking_list[:num_l]

        for u in u_list:
            # add u to the list of landmark nodes
            landmark_nodes.append(u)

            # get the distance from "u" to all other nodes
            shortest_path_lengths = nx.single_source_shortest_path_length(graph, u)
            sp_array = np.array(list(shortest_path_lengths.items()))

            # get an array of reached nodes
            reached_nodes = list(shortest_path_lengths.keys())
            recorded_distances  = list(shortest_path_lengths.values())

            df_u = pd.DataFrame()
            df_u["vertices"] = reached_nodes
            df_u[str(u)] = recorded_distances

            distance_mapping = distance_mapping.join(df_u.set_index("vertices"))

            # compute the average distance
            average_distance = sp_array[:, 1].mean()

            # get the nodes within average distance
            nodes_in_range = list(sp_array[np.where(sp_array[:,1] <= average_distance)[0], :][:,0])

            # remove the nodes that are within average distance
            updated_ranking_list = set(ranking_list) - set(nodes_in_range)
            ranking_list = list(updated_ranking_list)

    return distance_mapping


def approximate_distance_strategy1(s, t, distance_mapping):
    """
    This function approximates the distance between 2 nodes using the distance mapping, using the following formula:
            d(s,t) = Upper Bound = min(d(s,l_i) + (d(l_i,t))

    :param: s (int): an integer representing the source node
    :param: t (int): an integer representing the target node
    :param: distance_mapping (pandas.core.frame.DataFrame): pandas dataframe containing the distance mapping between nodes and
                                                            landmarks. This dataframe is the variable returned from one of the
                                                            landmark selection strategy functions

    :returns: list containing the following
        - dist (int): approximated distance between s and t
        - l_i (int): integer representing the landmark through which the shortest path passes
    """
    dist_upper_bound = distance_mapping.loc[[s, t], :].sum().min()
    dist_lower_bound = max(abs(distance_mapping.loc[s, :] - distance_mapping.loc[t, :]))

    return dist_upper_bound, dist_lower_bound



def compute_metrics_strategy1(graph, distance_mapping, N = 500):
    """
    This function randomly selects N source nodes and N target nodes, computes the actual distance, approximated
    distance, the approximation error and L/U ratio

    :param graph : the networkx graph.
    :param distance_mapping (pandas.core.frame.DataFrame): pandas dataframe containing the distance mapping between
                                                           nodes and landmarks. This dataframe is the variable returned
                                                           from one of the landmark selection strategy functions.

    :param N (int) : number of randomly selected source nodes and target nodes

    :return:
        - mean_approximation_error: the mean approximation error. Value is between 0 and 1.
        - mean_L_U: the ratio between the Lower Bound and Upper bound of the triangle inequalities. Value is
                    between 0 and 1.
    """
    test_nodes = np.random.choice(np.array(list(graph.nodes())), N*2)
    sources = test_nodes[:N]
    targets = test_nodes[N:]

    results = pd.DataFrame()
    upper_bounds = []
    lower_bounds = []
    actual_distances = []

    for i in range(len(sources)):
        s = sources[i]
        t = targets[i]

        if nx.has_path(graph, s, t) == False:
            np.random.seed(int(i) + 1000)
            [s, t] = np.random.choice(np.array(list(graph.nodes())), 2)

        actual_distances.append(nx.shortest_path_length(graph, source=s, target=t))
        upper, lower = approximate_distance_strategy1(s, t, distance_mapping)
        upper_bounds.append(upper)
        lower_bounds.append(lower)


    results["sources"] = sources
    results["targets"] = targets
    results["actual"] = actual_distances
    results["upper_bounds"] = upper_bounds
    results["lower_bounds"] = lower_bounds

    mean_approximation_error = np.mean((results["upper_bounds"] - results["actual"]) / results["upper_bounds"])
    mean_L_U = np.mean(results["lower_bounds"] / results["upper_bounds"])

    return mean_approximation_error, mean_L_U


def run_strategy1(paths, save_directory_strategy1):
    """
    This function is used to perform the pre-set experiments using the baseline strategy (i.e. random landmark selection)


    :param paths: dictionary containing, as keys, the name of the graphs and, as values, the paths towards the edgelists
                  in .csv format

    :param save_directory_strategy1: string denoting the path to the directory in which the results are saved
    """

    # this dictionary is used for strategy 1 and has the following key-value format:
    # key = total number of landmarks selected
    # value = list containing the number of landmarks to select at every iteration of the
    #         recursive_node_elimination() function
    levels = {
        40: [20, 10, 5, 5],
        60: [30, 15, 10, 5],
        80: [40, 25, 10, 5],
        100: [50, 30, 15, 5],
        120: [60, 40, 15, 5]
    }

    seeds = [50, 51, 52, 53, 54]

    # run experiments:
    for graph_key in paths.keys():
        # create variable in which to save the results
        results_df = pd.DataFrame()
        results_df["number_of_landmarks"] = list(levels.keys())

        print(f"Currently working on graph: {graph_key} ....")
        df = pd.read_csv(paths[graph_key]).loc[:, ["source", "target"]]
        graph = nx.from_pandas_edgelist(df, "source", "target", create_using=nx.Graph)
        print("created graph...")

        # make 2 variables for saving the average approximation error and the average L/U ratio
        avg_approx_error = []
        avg_L_U = []

        for k_num_l in levels.keys():
            print(f"Currently selecting {k_num_l} landmarks")

            tmp_approx = []
            tmp_LU = []

            for s in tqdm(seeds):
                np.random.seed(s)
                # obtain results for strategy 1: recursive node elimination
                distance_mapping = recursive_node_elimination(graph, num_landmarks=levels[k_num_l])

                mean_approximation_error, mean_L_U = compute_metrics_strategy1(graph, distance_mapping, N=500)

                tmp_approx.append(mean_approximation_error)
                tmp_LU.append(mean_L_U)

            avg_approx_error.append(np.mean(np.array(tmp_approx)))
            avg_L_U.append(np.mean(np.array(tmp_LU)))
            print(f"avg approx error: {np.mean(np.array(tmp_approx))}")

        results_df["approximation_error"] = avg_approx_error
        results_df["LU_ratio"] = avg_L_U

        save_path = save_directory_strategy1 + str(graph_key) + "_overall_results.csv"
        results_df.to_csv(save_path)

