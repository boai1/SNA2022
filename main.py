import numpy as np
import pandas as pd
import networkx as nx
import operator
from tqdm import tqdm
from networkx.algorithms.community import asyn_fluidc
import matplotlib.pyplot as plt
from operator import itemgetter
import math
from baseline_strategy import run_baseline_strategy
from strategy1 import run_strategy1
from strategy2 import run_strategy2



if __name__ == "__main__":

    # create a dictionary having
    #   - key = the name of the graph
    #   - value = path to the csv file containing the edges

    paths = {
        "dblp": r"./data/dblp_undirected.csv",
        "amazon": r"./data/amazon.csv",
        "youtube": r"./data/youtube.csv",
        "livemocha": r"./data/livemocha.csv",
        "wiki_page": r"./data/wiki_page_graph.csv",
    }

    # indicate the directories where to save the results of the experiments
    save_directory_baseline = r"./results/baseline/"
    save_directory_strategy1 = r"./results/strategy1/"
    save_directory_strategy2 = r"./results/strategy2/"

    # perform baseline experiments
    run_baseline_strategy(paths, save_directory_baseline)

    # perform the experiments for strategy 1
    run_strategy1(paths, save_directory_strategy1)

    # perform the experiments for strategy 2
    run_strategy2(paths, save_directory_strategy2)
