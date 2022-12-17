# SNACS 2022 Main Project Repository

This repository contains the code used for the Main Project of Group 53 (Bogdan Aioanei & Nan Sai) for the SNACS 2022 Course. 

In this paper we extend the landmark-based methods for point-to-point distance estimation in large networks introducing 2 new landmark selection strategies. Both methods presented in this paper approximate the distances between nodes using a selected subset of nodes, called landmarks. Thus, instead of determining the lengths of all possible paths between 2 nodes, the approximation method only looks at the paths that include the landmarks. The first approach forcibly selects landmarks that are spread out throughout the graph by not considering the nodes within a certain range of the already selected landmarks. The second method achieves this same feature by splitting the graph into communities and selecting a landmark in each community. The 2 strategies are compared against the random landmark selection strategy on 5 different networks.

