import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px

def plot_cluster(data, graph=None, cluster_col='cluster'):
    data = data.copy()
    if graph:
        data[cluster_col] = pd.Series(nx.get_node_attributes(graph, 'cluster'))
    n_cluster = data[cluster_col].nunique()
    fig = px.scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], color=data[cluster_col], color_continuous_scale='turbo', range_color=[0, 30])
    fig.update_layout(height=500, width=500)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(title=f'Number of clusters: {n_cluster}')
    fig.show()
    
    
def graph_sum(graph):
    df = pd.DataFrame({'cluster': pd.Series(nx.get_node_attributes(graph, 'cluster'))})
    cluster_count = df.groupby('cluster').size()
    return pd.DataFrame(cluster_count, columns=['count'])