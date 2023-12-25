import itertools

import numpy as np
import pandas as pd
import networkx as nx
import scipy
import pymetis
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

def knn_graph(df, n_neighbors=30, mode='distance', metric='euclidean', include_self=True):
    """get similarity graph from DataFrame using knn"""
    csr_matrix = kneighbors_graph(df.values, 
                                  n_neighbors=n_neighbors, 
                                  mode=mode, 
                                  metric=metric,
                                  include_self=include_self)
    distances = csr_matrix.data
    
    # transform distances to similarities 
    # # Approach 1: using gaussian kernel
    # sigma = np.median(distances)
    # similarities = np.exp(- distances ** 2 / (2 * sigma ** 2))
    
    # Approach 2: transform distances to similarities
    similarities = 1 / (distances + 1)

    # scale up the similarities to retain accuracy when converting to int
    similarities = similarities * 10000
    similarities = similarities.astype(int)

    csr_matrix.data = similarities

    graph = nx.from_scipy_sparse_array(csr_matrix)
    return graph


def knn_graph_f_distance_matrix(n_neighbors=30, dist_mat=None, dist_to_sim=None):
    """create knn graph from precomputed distance matrix"""
    graph = nx.Graph()

    # add node
    graph.add_nodes_from(range(dist_mat.shape[0]))

    # add edge with similarity as edge weight
    for i in range(dist_mat.shape[0]):
        distances = list(enumerate(dist_mat[i]))
        distances.sort(key=lambda x: x[1])
        neighbors = distances[1:n_neighbors+1]
        
        for neighbor in neighbors:
            j = neighbor[0]  # index or neighbor
            weight = neighbor[1]
            # transform distance to similarity
            weight = 1 / (1 + weight) if dist_to_sim is None else dist_to_sim(weight)
            weight = int(weight * 10000)
            graph.add_edge(i, j, weight=weight)
            
    return graph


def ensemble_similarity_graph(df):
    """get similarity graph from DataFrame using ensemble clustering"""
    df = df.copy()
    df['cluster'] = KMeans(n_clusters=30, n_init='auto').fit(df.iloc[:, :2]).labels_
    cluster_res = []
    for _ in range(50):
        cluster_vec = KMeans(n_clusters=30, n_init='auto').fit(df.iloc[:, :2]).labels_
        # make cluster_vec into one-hot matrix
        membership_mat = pd.get_dummies(cluster_vec).astype(int).values
        cluster_res.append(membership_mat)
    concat_mat = np.concatenate(cluster_res, axis=1)
    csr_matrix = scipy.sparse.csr_matrix(concat_mat)
    similarity_mat = csr_matrix.dot(csr_matrix.T) / 50
    
    # scale up the similarities to retain accuracy when converting to int
    similarity_mat = similarity_mat * 10000
    similarity_mat = similarity_mat.astype(int)
    
    graph = nx.from_scipy_sparse_array(similarity_mat)
    return graph


def part_graph(graph, n_parts=2):
    """Partition a networkx graph into nparts using metis
    
    Args:
        graph (networkx.Graph): the graph to partition, where weights must be integer
        nparts (int): the number of partitions
    
    Returns:
        n_cuts (int): the number of cuts
        membership (list): the cluster each node belongs to
    """
    
    csr = nx.to_scipy_sparse_array(graph)
    xadj = csr.indptr
    adjncy = csr.indices
    eweights = csr.data  # Has to be integer
    xadj, adjncy, eweights = xadj.tolist(), adjncy.tolist(), eweights.tolist()
    n_cuts, membership = pymetis.part_graph(nparts=n_parts, xadj=xadj, adjncy=adjncy, eweights=eweights)
    return n_cuts, membership


def partition(graph, exclude_cluster=None, initialize=True):
    """Find the largest cluster and partition it into two subclusters

    Args:
        graph (networkx.Graph)
        exclude_cluster (list, optional): the clusters to exclude when finding the largest cluster
        initialize (boolean): whether to initialize nodes without cluster
    """
    cluster_idx = pd.DataFrame({'cluster': nx.get_node_attributes(graph, 'cluster')})
    int_cluster = [i for i in cluster_idx['cluster'].unique() if isinstance(i, int) or isinstance(i, np.int64)]
    
    if initialize:
        # if exist nodes without cluster, initialize these nodes
        if len(nx.get_node_attributes(graph, 'cluster')) != len(graph):
            node_wo_cluster = list(set(graph.nodes()) - set(nx.get_node_attributes(graph, 'cluster').keys()))
            init_cluster = min(int_cluster) - 1 if int_cluster else -1
            
            nx.set_node_attributes(graph, {node: init_cluster for node in node_wo_cluster}, 'cluster')
            print(f'Eixst nodes without cluster. Initialize {len(node_wo_cluster)} nodes to cluster {init_cluster}')
            cluster_idx = pd.DataFrame({'cluster': nx.get_node_attributes(graph, 'cluster')})
            int_cluster = [i for i in cluster_idx['cluster'].unique() if isinstance(i, int)]
    
    new_cluster_idx = max(int_cluster) + 1 if int_cluster else 1
    
    # find the largest cluster that is not excluded
    cluster_count = cluster_idx.groupby('cluster').size()
    if exclude_cluster is not None:
        cluster_count = cluster_count.drop(exclude_cluster, errors='ignore')
    cluster_count.sort_values(ascending=False, inplace=True)
    largest_cluster = cluster_count.index[0]
    sub_graph = graph.subgraph(cluster_idx[cluster_idx['cluster'] == largest_cluster].index.values)
    
    # partition the largest cluster into two
    _, membership = part_graph(sub_graph, n_parts=2)
    
    # assign new cluster idx to nodes in the second partition
    for node_idx, cluster_idx in zip(sub_graph.nodes(), membership):
        if cluster_idx == 1:
            graph.nodes[node_idx]['cluster'] = new_cluster_idx
            

def get_cluster_f_graph(graph, cluster_idx):
    """Get the nodes in a cluster as a subgraph"""
    selected_nodes = [node for node, attr in graph.nodes(data=True) if attr['cluster'] == cluster_idx]
    subgraph = graph.subgraph(selected_nodes)
    return subgraph


def min_cut_weights(graph, cluster_idx):
    """Get the weights of the edges that are cut when partitioning a cluster into two roughly equal parts"""
    subgraph = get_cluster_f_graph(graph, cluster_idx)
    _, partition = part_graph(subgraph, n_parts=2)
    sub_cluster1 = [node for node, cluster in zip(subgraph.nodes(), partition) if cluster == 0]
    sub_cluster2 = [node for node, cluster in zip(subgraph.nodes(), partition) if cluster == 1]
    edge_cut = nx.edge_boundary(subgraph, sub_cluster1, sub_cluster2)
    weights_of_edge_cut = [subgraph.edges[edge]['weight'] for edge in edge_cut]
    return weights_of_edge_cut


def similarity_between_clusters(graph, cluster_idx1, cluster_idx2, min_cut_weights1=None, min_cut_weights2=None, alpha=2):
    """
    Calculates the similarity between two clusters in a graph.

    Args:
        graph: NetworkX graph object
        cluster_idx1: Index of the first cluster
        cluster_idx2: Index of the second cluster
        min_cut_weights1: List of weights for the minimum cut of cluster_idx1 (optional)
        min_cut_weights2: List of weights for the minimum cut of cluster_idx2 (optional)
        alpha: the relative importance of the inter-connectivity

    Return:
        similarity: Similarity between the two clusters
    """
    cluster1 = get_cluster_f_graph(graph, cluster_idx1)
    cluster2 = get_cluster_f_graph(graph, cluster_idx2)
    edge_cut = nx.edge_boundary(graph, cluster1, cluster2)
    weights_of_edge_cut = [graph.edges[edge]['weight'] for edge in edge_cut]
    min_cut_weights1 = min_cut_weights(graph, cluster_idx1) if min_cut_weights1 is None else min_cut_weights1
    min_cut_weights2 = min_cut_weights(graph, cluster_idx2) if min_cut_weights2 is None else min_cut_weights2
    
    # internal interconnectivity
    ii1, ii2 = np.sum(min_cut_weights1), np.sum(min_cut_weights2)
    # internal inter-connectivity
    ic1 = np.mean(min_cut_weights1) if min_cut_weights1 else 0
    ic2 = np.mean(min_cut_weights2) if min_cut_weights2 else 0
    size1, size2 = len(cluster1), len(cluster2)
    
    # absolute inter-connectivity and closeness
    ai = np.sum(weights_of_edge_cut)
    ac = np.mean(weights_of_edge_cut) if weights_of_edge_cut else 0
    
    # relative inter-connectivity and closeness
    denominator1 = np.abs(ii1) + np.abs(ii2)
    ri = (2 * ai) / (np.abs(ii1) + np.abs(ii2)) if denominator1 else 0
    denominator2 = (size1 / (size1 + size2)) * ic1 + (size2 / (size1 + size2)) * ic2
    rc = ac / ((size1 / (size1 + size2)) * ic1 + (size2 / (size1 + size2)) * ic2) if denominator2 else 0
    
    # similarity
    similarity = ri * rc ** alpha
    
    return similarity


def find_closest_clusters(graph, alpha=2, cl_mat=None):
    """Find the closest clusters that do not have cannot-link pairs
    
    Args:
        graph (networkx.Graph): the graph to partition
        alpha (float): the parameter to control the importance of internal interconnectivity and inter-cluster connectivity
        cl_mat (np.ndarray): cannot-link matrix, showing whether two instances have cannot-link constraints
    """
    cluster_idxs = pd.DataFrame({'cluster': pd.Series(nx.get_node_attributes(graph, 'cluster'))})
    
    # Precompute the min-cut weights for each cluster
    min_cut_weights_dict = {}
    for cluster_idx in cluster_idxs['cluster'].unique():
        min_cut_weights_dict[cluster_idx] = min_cut_weights(graph, cluster_idx)
        
    if cl_mat is not None:
        # create a matrix showing whether two clusters have cannot-link pairs
        cluster_cl_mat = pd.DataFrame(cl_mat, index=cluster_idxs['cluster'], columns=cluster_idxs['cluster'])
        cannot_link_between_clusters = cluster_cl_mat.groupby(level=0).any().T.groupby(level=0).any()
            
        cluster_pairs = itertools.combinations(cluster_idxs['cluster'].unique(), 2)
        similarities = []
        for cluster_idx1, cluster_idx2 in cluster_pairs:
            # if there are cannot-link pairs between the two clusters, return similarity 0
            if cannot_link_between_clusters.loc[cluster_idx1, cluster_idx2]:
                similarity = 0
            else:
                similarity = similarity_between_clusters(graph, cluster_idx1, cluster_idx2, 
                                                         min_cut_weights_dict[cluster_idx1], 
                                                         min_cut_weights_dict[cluster_idx2], 
                                                         alpha=alpha)
            similarities.append((cluster_idx1, cluster_idx2, similarity))
    else:
        cluster_pairs = itertools.combinations(cluster_idxs['cluster'].unique(), 2)
        similarities = []
        
        for cluster_idx1, cluster_idx2 in cluster_pairs:
            similarity = similarity_between_clusters(graph, cluster_idx1, cluster_idx2, 
                                                     min_cut_weights_dict[cluster_idx1], 
                                                     min_cut_weights_dict[cluster_idx2], 
                                                     alpha=alpha)
            similarities.append((cluster_idx1, cluster_idx2, similarity))
        
    similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
    closest_pair = similarities[0][0], similarities[0][1]
    return closest_pair


def merge_closest_clusters(graph, alpha=2, cl_mat=None):
    closest_pair = find_closest_clusters(graph, alpha=alpha, cl_mat=cl_mat)
    cluster_dict = nx.get_node_attributes(graph, 'cluster')
    nodes_to_merge = [node for node, cluster in cluster_dict.items() if cluster == closest_pair[1]]
    new_cluster = {node: {'cluster': closest_pair[0]} for node in nodes_to_merge}
    nx.set_node_attributes(graph, new_cluster)
    return closest_pair