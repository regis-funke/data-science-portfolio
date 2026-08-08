# European Road Network Analysis for Logistics Center Location

## Overview
This project aims to analyze a European road network to identify potential locations for logistics centers. The network, sourced from networkrepository.com, represents highway access points (nodes) and connecting highways (edges).

## Project Description
The project includes several key analyses:
1. **Graph Construction**: The road network is modeled as a graph with nodes representing highway access points and edges representing highways.
2. **Graph Analysis**: Exploratory data analysis of the graph's properties, including its density, diameter, average shortest path length, and clustering coefficients.
3. **Centrality Measures**: Evaluation of nodes based on betweenness and closeness centrality to identify critical hubs in the network.
4. **Community Detection**: Using the Louvain method to discover well-connected regions within the network, which are potential locations for logistics centers.
5. **Link Prediction**: Predicting future road construction using node names and edge embeddings with a RandomForestClassifier.

## Methodology
- **Data Preparation**: The dataset from networkrepository.com, containing 1039 nodes and 1305 edges, is used to construct the graph.
- **Graph Analysis**: Network properties are analyzed using NetworkX, including measures like density, transitivity, and degree assortativity.
- **Centrality Analysis**: Betweenness and closeness centrality measures are computed to identify important nodes in the network.
- **Community Detection**: The Louvain method is applied to partition the network into communities, with each community potentially representing a country or region.
- **Link Prediction**: The RandomForestClassifier is trained on node names and edge embeddings to predict the likelihood of future road construction between points.

## Key Findings
- The road network exhibits a low density, indicating it is less densely connected compared to other types of networks like social networks.
- The network's diameter and average shortest path length are significantly larger than those in a random graph, suggesting more isolated areas.
- Nodes with high betweenness and closeness centrality were identified as potential locations for logistics centers.
- The Louvain method revealed 21-22 communities within the network, correlating roughly with the number of EU member states.

## Technologies Used
- Python for data analysis and machine learning.
- NetworkX for graph analysis and community detection.
- RandomForestClassifier for link prediction.
- Node2Vec for generating node embeddings.
- Matplotlib and community packages for visualization and community detection.

---

[← Portfolio index](../../../README.md)
