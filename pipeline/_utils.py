"""
Functions for labeling and encoding chemical characters like Compound SMILES and atom string, refer to
https://github.com/hkmztrk/DeepDTA and https://github.com/thinng/GraphDTA.
"""

import logging
from tkinter.font import names

import numpy as np
from rdkit import Chem
import os
import random
import numpy as np
import torch


CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}


cluster_mapping = {
    "G": 1, "A": 1, "V": 1, "L": 1, "I": 1,  # Aliphatic
    "F": 2, "Y": 2, "W": 2,   # Aromatic
    "C": 3, "M": 3, "P": 3, "S": 3, "T": 3, "N": 3, "Q": 3,  # Neutral
    "H": 4, "K": 4, "R": 4, # Positively charged
    "D": 5, "E": 5,  # Negatively charged
    "B": 6, "X": 6, "O": 6, "U":6, "Z": 6, # Special Cases
}

def integer_label_protein(sequence, max_length):
    """
    Integer encoding for protein string sequence.

    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding


def one_hot_protein(sequence, max_length):
    """
    One-hot encoding for protein string sequence.

    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    unique_proteins = len(set(CHARPROTSET.values()))
    encoding = np.zeros((max_length, unique_proteins))

    for idx, letter in enumerate(sequence[:max_length]):
        letter = letter.upper()
        encoding[idx, (CHARPROTSET[letter]-1)] = 1
    return encoding


def one_hot_cluster(sequence, max_length):
    """
    One-hot encoding for protein string sequence with cluster mapping.

    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    unique_clusters = len(set(cluster_mapping.values()))
    encoding = np.zeros((max_length, unique_clusters))

    for idx, letter in enumerate(sequence[:max_length]):
        letter = letter.upper()
        encoding[idx, (cluster_mapping[letter]-1)] = 1
    return encoding



def integer_label_cluster(sequence, max_length):
    """
    Integer encoding for protein string sequence with cluster mapping.

    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        letter = letter.upper()
        encoding[idx] = cluster_mapping[letter]
    return encoding


# ----- With only positions in selected index -----

def one_hot_protein_selected(sequence, max_length, selected_index):
    """
    One-hot encoding for protein string sequence with selected index.

    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
        selected_index: Selected index to encode.
    """
    unique_proteins = len(set(CHARPROTSET.values()))
    encoding = np.zeros((max_length, unique_proteins))

    for idx, letter in enumerate(sequence[:max_length]):
        if idx in selected_index:
            letter = letter.upper()
            encoding[idx, (CHARPROTSET[letter] - 1)] = 1
    return encoding




"""Setting seed for reproducibility"""
# Results can be software/hardware-dependent
# Exactly reproduciable results are expected only on the same software and hardware
def set_seed(seed):
    """Sets the seed for generating random numbers to get (as) reproducible (as possible) results.

    The CuDNN options are set according to the official PyTorch guidance on reproducibility:
    https://pytorch.org/docs/stable/notes/randomness.html. Another references are
    https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848/6
    https://pytorch.org/docs/stable/cuda.html#torch.cuda.manual_seed
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/utils.py#L58

    Args:
        seed (int, optional): The desired seed. Defaults to 1000.
    """

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
    # 4. Set `pytorch` pseudo-random generator at a fixed value
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)



def result_holder():
    sum_of_scores = {}
    sum_of_scores.update({
        "running_time": None,

        "accuracy": [],
        "auroc": [],
        "recall": [],
        "specificity": [],
        "precision": [],
        "npv": [],
    })

    misc = {
        "model_weights": [],
        "train_index": [],
        "test_index": [],
    }

    hyperparameters = {}

    return sum_of_scores, misc, hyperparameters