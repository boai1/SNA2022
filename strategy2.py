import numpy as np
import networkx as nx
import pandas as pd
from networkx.algorithms.community import asyn_fluidc
from operator import itemgetter
import operator
from tqdm import tqdm


def community_partition(graph, k=50, seed=123):
    """
    This function implements strategy 2 using Fluid Community detection algorithm.

    :param graph (nx.Graph): the networkx graph created from the edgelist.
    :param k (int): indicates the number of communities that need to be found. It also indicates the number of landmarks
                    (since there will be 1 landmark/community).
    :param seed (int): the random seed used in the community detection algorithm.

    :returns:
      - distance_mapping (pandas.core.frame.DataFrame): a pandas DataFrame having as index column the list of all nodes in
                                                        the graph. Each column other than the index column is denoted by a
                                                        landmark and contains the distance from the landmark to every other node.
                                                        If a node is not reachable by a landmark, the value in the cell will
                                                        be infinity.
    """

    # obtain the communities
    landmark_communities = list(asyn_fluidc(graph, k=k, seed=seed))

    # obtain the degree centrality of all nodes in the graph
    degr_centrality_ranking = dict(
        sorted(nx.degree_centrality(graph).items(), key=operator.itemgetter(1), reverse=True))

    degr_centrality_of_communities = {}

    # create a pandas dataframe in which to store the node-to-landmark distances
    distance_mapping = pd.DataFrame()
    distance_mapping["vertices"] = list(graph.nodes())
    distance_mapping = distance_mapping.set_index("vertices")

    all_landmarks = {}

    for i in range(len(landmark_communities)):
        # get the nodes in community i
        d = {key: degr_centrality_ranking.get(key) for key in landmark_communities[i]}

        # rank the nodes in the community based on degree centrality and pick the top node
        landmark_i = list(dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:1]).keys())[0]

        # add the node to the list of landmarks
        all_landmarks[landmark_i] = list(d.keys())

    for u, community in list(all_landmarks.items()):
        # compute the distance from this landmark to all reachable nodes
        reached_nodes = [m for m in community] + list(all_landmarks.keys())
        recorded_distances = [nx.shortest_path_length(graph, source=u, target=t) for t in reached_nodes]

        # create a dataframe in which to save the node-to-landmark distances
        df_i = pd.DataFrame()
        df_i["vertices"] = reached_nodes
        df_i[str(u)] = recorded_distances

        df_i = df_i[df_i["vertices"].isin(reached_nodes)]

        # add the results-per-community to the larger dataframe containing results-per-graph
        distance_mapping = distance_mapping.join(df_i.set_index("vertices"))

    # replace NaN values to infinity
    distance_mapping.fillna(np.inf, inplace=True)

    return distance_mapping, all_landmarks


def approximate_distance_strategy2(s, t, distance_mapping, all_landmarks):
    """
    This function is used to approximate the distance between 2 nodes using the pre-computed distance mapping of
    strategy 2: Community Partition
    :param s (int): integer representing the source node
    :param t (int): integer representing the target node
    :param distance_mapping (pandas dataframe): the distance mapping variable
    :param all_landmarks: dictionary having as keys: the landmarks, and as values: a list of all the nodes in their
           respective community

    :return: approx_dist (int): integer representing the approximated distance
    """
    for l, comm in all_landmarks.items():
        if s in comm:
            landmark_s = l
        if t in comm:
            landmark_t = l

    dist_s_to_l = distance_mapping.loc[s, str(landmark_s)]
    dist_l_to_l = list(distance_mapping.loc[landmark_s, str(landmark_t)])[0]
    dist_l_to_t = distance_mapping.loc[t, str(landmark_t)]

    approx_dist = dist_s_to_l + dist_l_to_l + dist_l_to_t

    return approx_dist

def compute_metrics_strategy2(graph, distance_mapping, all_landmarks, N = 500):
    """
    This function randomly selects N source nodes and N target nodes, computes the actual distance and the approximated
    distance

    :param graph : the networkx graph.
    :param distance_mapping (pandas.core.frame.DataFrame): pandas dataframe containing the distance mapping between
                                                           nodes and landmarks. This dataframe is the variable returned
                                                           from one of the landmark selection strategy functions.

    :param N (int) : number of randomly selected source nodes and target nodes

    :return: mean_approximation_error(int): the mean approximation error. Value is between 0 and 1.
    """
    # randomly select 500 pairs of nodes
    test_nodes = np.random.choice(np.array(list(graph.nodes())), N * 2)
    sources = test_nodes[:N]
    targets = test_nodes[N:]

    results = pd.DataFrame()
    approx_distances = []
    actual_distances = []

    for i in range(len(sources)):
        s = sources[i]
        t = targets[i]

        # if there isn't a path connecting the s and t nodes, select other nodes
        if nx.has_path(graph, s, t) == False:
            np.random.seed(int(i) + 1000)
            [s, t] = np.random.choice(np.array(list(graph.nodes())), 2)

        actual_d = nx.shortest_path_length(graph, source=s, target=t)
        actual_distances.append(int(actual_d))
        approx_d = approximate_distance_strategy2(s, t, distance_mapping, all_landmarks)

        if type(approx_d) == pd.core.series.Series:
            approx_distances.append(list(approx_d)[0])
        else:
            approx_distances.append(approx_d)

    # save the results
    results["sources"] = sources
    results["targets"] = targets
    results["actual"] = actual_distances
    results["approx"] = approx_distances

    if results.isin([np.inf, -np.inf, np.nan]).sum().sum() == 0:
        err = []
        for i in results.index:
            act = results.loc[i, "actual"]
            app = results.loc[i, "approx"]
            err.append((app - act) / app)

        mean_approximation_error = np.mean(np.array(err))

    else:
        mean_approximation_error = np.inf

    return mean_approximation_error


def get_n_edges_in_component(G, node_list):
    """
    This function extracts, from the main graph, the edges formed between a certain set of nodes.

    :param G: networkx graph object representing the original main grap
    :param node_list: list of nodes

    :return: list of edges formed between the nodes in the node_list variable
    """

    G_edge_list = list(G.edges())

    filtered_edges = list(filter(lambda edge: (edge[0] in node_list and edge[1] in node_list), G_edge_list))

    return filtered_edges


def run_strategy2(paths, save_directory_strategy2):
    """
    This function is used to perform the pre-set experiments using the baseline strategy (i.e. random landmark selection)


    :param paths: dictionary containing, as keys, the name of the graphs and, as values, the paths towards the edgelists
                  in .csv format

    :param save_directory_strategy2: string denoting the path to the directory in which the results are saved
    """
    # input the number of landmarks used for the experiments
    n_landmarks = [40, 60, 80, 100, 120]

    # set of seeds used
    seeds = [4, 0, 1, 2, 3]

    for graph_key in paths.keys():
        # create variable in which to save the results
        results_df = pd.DataFrame()
        results_df["number_of_landmarks"] = n_landmarks

        print(f"Currently working on graph: {graph_key} ....")
        df = pd.read_csv(paths[graph_key]).loc[:, ["source", "target"]]
        graph = nx.from_pandas_edgelist(df, "source", "target", create_using=nx.Graph)

        # get the largest connected component
        largest_component= max(nx.connected_components(graph), key=len)

        # extract the edges from the largest connected component
        largest_component_edges = get_n_edges_in_component(graph, largest_component)

        # Create a new graph from the edges of the connected component
        lcc = nx.Graph()
        lcc.add_edges_from(largest_component_edges)

        print("created graph...")

        avg_approx_error = []

        for n_l in n_landmarks:
            print(f"currently working on: {n_l} landmarks")
            tmp_approx = []
            # obtain results for strategy 2: community node partition
            distance_mapping, all_landmarks = community_partition(lcc, k=n_l, seed=13)

            for s in tqdm(seeds):
                np.random.seed(s)
                mean_approximation_error = compute_metrics_strategy2(lcc, distance_mapping, all_landmarks, N = 500)
                tmp_approx.append(mean_approximation_error)

            if np.size(np.where(tmp_approx == np.inf)[0]) == 0:
                avg_approx_error.append(np.mean(np.array(tmp_approx)))
                print(f"avg approx error: {np.mean(np.array(tmp_approx))}")
            else:
                avg_approx_error.append(np.inf)

        results_df["approximation_error"] = avg_approx_error

        save_path = save_directory_strategy2 + str(graph_key) + "_overall_results.csv"
        results_df.to_csv(save_path)