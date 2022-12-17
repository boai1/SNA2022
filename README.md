# SNACS 2022 Main Project Repository

This repository contains the code used for the Main Project of Group 53 (Bogdan Aioanei & Nan Sai) for the SNACS 2022 Course. 

In this paper we extend the landmark-based methods for point-to-point distance estimation in large networks introducing 2 new landmark selection strategies. Both methods presented in this paper approximate the distances between nodes using a selected subset of nodes, called landmarks. Thus, instead of determining the lengths of all possible paths between 2 nodes, the approximation method only looks at the paths that include the landmarks. The first approach forcibly selects landmarks that are spread out throughout the graph by not considering the nodes within a certain range of the already selected landmarks. The second method achieves this same feature by splitting the graph into communities and selecting a landmark in each community. The 2 strategies are compared against the random landmark selection strategy on 5 different networks.


# Usage
In order to run the main.py file, do the following:
1) download the edgelists for the 5 networks in .csv format using the following link:  
  https://drive.google.com/drive/folders/1UPAU1kUnXTO-GdP7w_bdkp8TMuAApUX_?usp=sharing
  
2) add the .csv files to the following directory: SNACS2022/data/ 

   Note: This directory already contains the amazon.csv and dblp_undirected.csv graph edgelists. The code can be run using only these 2 graphs, but at some point it will return an error due to the absence of the other graphs, which were too big for github. 
   
3) run main.py

