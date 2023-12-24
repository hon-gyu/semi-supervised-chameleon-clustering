import numpy as np
import pandas as pd
import networkx as nx
from rich.progress import Progress, track
from graph import partition, merge_closest_clusters

def partition_phase(graph, n_cluster_final=30, exclude_cluster=None):
    """Partition the graph until the number of clusters reaches n_cluster_final

    Args:
        graph (networkx.Graph)
        n_cluster_final (int, optional): the number of clusters to partition to.
        exclude_cluster (list, optional): the clusters to exclude when finding the largest cluster
    """
    n_cluster = pd.Series(nx.get_node_attributes(graph, 'cluster')).nunique()
    
    if n_cluster >= n_cluster_final:
        print(f'Number of clusters {n_cluster} has already')
        raise ValueError('n_cluster_final must be larger than the number of clusters in the graph')
    
    while n_cluster < n_cluster_final:
        partition(graph, exclude_cluster=exclude_cluster)
        n_cluster = pd.Series(nx.get_node_attributes(graph, 'cluster')).nunique()
        # print(f'Number of clusters: {n_cluster}')


def merge_phase(graph, n_cluster_final, alpha=2, cl_mat=None):
    """Merge clusters until the number of clusters reaches n_cluster_final

    Args:
        graph (networkx.Graph): the graph to partition
        n_cluster_final (int): the number of clusters to partition to
        alpha (float): the parameter to control the importance of internal interconnectivity and inter-cluster connectivity
        cl_mat (np.ndarray): cannot-link matrix, showing whether two instances have cannot-link constraints
    """

    cluster_idxs = pd.DataFrame({'cluster': pd.Series(nx.get_node_attributes(graph, 'cluster'))})
    n_cluster = cluster_idxs['cluster'].nunique()
    if n_cluster <= n_cluster_final:
        raise ValueError('No need to merge')
    n_merges = n_cluster - n_cluster_final

    with Progress() as p:
        task1 = p.add_task("[red]Processing...", total=n_merges)
        
        while n_cluster > n_cluster_final:
            merged_pairs = merge_closest_clusters(graph, alpha=alpha, cl_mat=cl_mat)
            n_cluster -= 1
            print(f'Number of clusters: {n_cluster}, Merged pairs: {merged_pairs}')
            
            p.update(task1, advance=1)