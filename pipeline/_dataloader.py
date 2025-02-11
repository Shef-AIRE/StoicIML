import torch.utils.data as data
from pipeline._utils import integer_label_protein, one_hot_protein, one_hot_cluster, integer_label_cluster, one_hot_protein_selected
import os
import pickle
import numpy as np

def to_embedding(df, max_length):
    """
    Convert protein sequence to integer encoding.
    """
    index = df.index.tolist()
    v_p = df['Protein sequence'].apply(lambda seq: integer_label_protein(seq, max_length)).tolist()
    stoichiometry = df['Stoichiometry'].tolist()
    return index, v_p, stoichiometry

def to_embedding_one_hot_protein(df, max_length):
    """
    Convert protein sequence to one-hot encoding.
    """
    index = df.index.tolist()
    v_p = df['Protein sequence'].apply(lambda seq: one_hot_protein(seq, max_length)).tolist()
    # reshape v_p
    v_p = np.array(v_p)
    n_samples, max_length, unique_clusters = v_p.shape
    v_p_flattened = v_p.reshape(n_samples, max_length * unique_clusters)

    stoichiometry = df['Stoichiometry'].tolist()
    return index, v_p_flattened, stoichiometry


def to_embedding_one_hot_cluster(df, max_length):
    """
    Convert protein sequence to one-hot encoding with cluster mapping.
    """
    index = df.index.tolist()

    v_p = df['Protein sequence'].apply(lambda seq: one_hot_cluster(seq, max_length)).tolist()
    # reshape v_p
    v_p = np.array(v_p)
    n_samples, max_length, unique_clusters = v_p.shape
    v_p_flattened = v_p.reshape(n_samples, max_length * unique_clusters)

    stoichiometry = df['Stoichiometry'].tolist()
    return index, v_p_flattened, stoichiometry


def to_embedding_integer_label_cluster(df, max_length):
    """
    Convert protein sequence to integer encoding with cluster mapping.
    """
    index = df.index.tolist()
    v_p = df['Protein sequence'].apply(lambda seq: integer_label_cluster(seq, max_length)).tolist()
    stoichiometry = df['Stoichiometry'].tolist()
    return index, v_p, stoichiometry



# ----- With only positions in selected index -----
def to_embedding_one_hot_protein_selected(df, max_length, selected_index):
    """
    Convert protein sequence to one-hot encoding.
    """
    index = df.index.tolist()
    v_p = df['Protein sequence'].apply(lambda seq: one_hot_protein_selected(seq, max_length, selected_index)).tolist()
    # reshape v_p
    v_p = np.array(v_p)
    n_samples, max_length, unique_clusters = v_p.shape
    v_p_flattened = v_p.reshape(n_samples, max_length * unique_clusters)

    stoichiometry = df['Stoichiometry'].tolist()
    return index, v_p_flattened, stoichiometry
