import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

def generate_synthetic_data_with_hierarchy(n_samples=10_000, n_features=5, n_clusters=6, random_state=4, cluster_std=2):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state, cluster_std=cluster_std)
    
    # Add hierarchy to the clusters
    data = pd.DataFrame(X, columns=[f"feat{i}" for i in range(1, n_features+1)])
    data = pd.concat([data, pd.Series(y, name='true_clst_l3')], axis=1)

    data['true_clst_l1'] = data['true_clst_l3'].apply(lambda x: (0, 1, 2, 3) if x in [0, 1, 2, 3] else (4, 5))
    def l2(x):
        if x in [0, 1]:
            return (0, 1)
        elif x in [2, 3]:
            return (2, 3)
        elif x in [4, 5]:
            return (4, 5)
    data['true_clst_l2'] = data['true_clst_l3'].apply(l2)
    data = data[['feat1', 'feat2', 'feat3', 'feat4', 'feat5', 'true_clst_l1',
        'true_clst_l2', 'true_clst_l3']].copy()
    data['true_clst'] = data['true_clst_l1'].astype(str) + '-' + data['true_clst_l2'].astype(str) + '-' + data['true_clst_l3'].astype(str)
    
    return data


def known_tag(x):
    """mapping true cluster to known tag
    60% is missing
    20% is correct at l1 level
    10% is correct at l2 level
    10% is correct at l3 level
    """
    coin = np.random.random()
    l1, l2, l3 = 'nan', 'nan', 'nan'
    if coin <= 0.6:
        pass
    elif coin <= 0.8:
        l1 = str(x['true_clst_l1'])
    elif coin <= 0.9:
        l1 = str(x['true_clst_l1'])
        l2 = str(x['true_clst_l2'])
    else:
        l1 = str(x['true_clst_l1'])
        l2 = str(x['true_clst_l2'])
        l3 = str(x['true_clst_l3'])
    tag = l1 + '-' + l2 + '-' + l3
    return tag


def create_synthetic_paritally_known_label(data):
    data = data.copy()
    data['known_tag'] = data.apply(known_tag, axis=1)
    data[['known_tag_l1', 'known_tag_l2', 'known_tag_l3']] = data['known_tag'].str.split('-', expand=True)
    data[['known_tag_l1', 'known_tag_l2', 'known_tag_l3']] = data[['known_tag_l1', 'known_tag_l2', 'known_tag_l3']].replace('nan', np.nan)
    return data