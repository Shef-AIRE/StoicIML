import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pipeline._utils import CHARPROTSET, cluster_mapping
from openpyxl import Workbook
from openpyxl.styles import PatternFill

from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from sklearn.feature_selection import VarianceThreshold

from matplotlib.ticker import FuncFormatter


def get_positional_weights(parent_dir):
    with open(os.path.join(parent_dir, "misc.pkl"), "rb") as file:
        misc = pickle.load(file)
        model_weights = misc["model_weights"]


    positional_weights = []
    # unpack the weights
    for model_number, weight in enumerate(model_weights):
        weight = weight.reshape(1426,-1)    # shape: (max_length, #AAs)
        # ---- 1. Take absolute value of all ----
        weight = np.abs(weight)
        # ---- 2. Sum all the weights for each position ----
        weight = weight.sum(axis=1)    # shape: (max_length,)
        positional_weights.append(weight)

    # ---- 3. Get the mean and std of the positional weights ----
    positional_weights = np.array(positional_weights)   # shape: (#models, max_length)
    mean_pw = positional_weights.mean(axis=0)   # shape: (max_length,)
    std_pw = positional_weights.std(axis=0)   # shape: (max_length,)
    return positional_weights, mean_pw, std_pw

# ================= get positional weights =================
# parent_dir = "main-exp-result/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9"
# positional_weights, mean_pw, std_pw = get_positional_weights(parent_dir)
# print(mean_pw)
# # save mean_pw to df in a column and save to excel
# df = pd.DataFrame(mean_pw)
# df.to_excel(os.path.join(parent_dir, "mean_positional_weights.xlsx"), index=False)
# ========================================================
# indices = np.load("main-exp-result/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9/capped_indices_perc6.npy")
# # 0-indexed to 1-indexed
# indices += 1
# print(indices)
# print(np.sort(indices))
# # create a df with the length of 129, for each position, if it is in the indices, then 'yes', else 'no', and save as a column to excel
# df = pd.DataFrame(columns=["Position", "Capped"])
# df["Position"] = range(1, 1427)
# df["Capped"] = "no"
# df.loc[df["Position"].isin(indices), "Capped"] = "yes"
# df.to_excel("main-exp-result/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9/capped_positions.xlsx", index=False)
# ========================================================

def get_capped_indices(mean_pw, std_pw, threshold):
    # ---- number of indices to get ----
    n = int(len(mean_pw) * threshold)
    print(f"Getting top {n} indices")

    # sort the mean_pw from highest to lowest
    sorted_indices = np.argsort(mean_pw)[::-1]
    capped_indices = sorted_indices[:n]
    print(capped_indices)

    # print the weights of the capped indices
    print(mean_pw[capped_indices])
    print(mean_pw[:10])
    return capped_indices

# ================= get capped indices =================
# parent_dir = "main-exp-result/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9"
# threshold = 0.06
# positional_weights, mean_pw, std_pw = get_positional_weights(parent_dir)
# capped_indices = get_capped_indices(mean_pw, std_pw, threshold)
# np.save(os.path.join(parent_dir, f"capped_indices_perc{int(threshold*100)}.npy"), capped_indices)
# ====================================================

def get_laplacian_indices(positional_weights, threshold):
    W = construct_W.construct_W(positional_weights, method='heat_kernel', t=1)
    score = lap_score.lap_score(positional_weights, W=W)
    sorted_indices = np.argsort(score) # from lowest -- as low Laplacian Score are preferred

    n = int(len(score) * threshold)
    capped_indices = sorted_indices[:n]
    print(f"Getting top {n} indices")
    return capped_indices

# ================= get laplacian indices =================
# parent_dir = "main-exp-result/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9"
# positional_weights, mean_pw, std_pw = get_positional_weights(parent_dir)
# threshold = 0.14
# capped_indices = get_laplacian_indices(positional_weights, threshold)
# np.save(os.path.join(parent_dir, f"laplacian_indices_perc{int(threshold*100)}.npy"), capped_indices)
# ========================================================

def get_simple_variance_indices(positional_weights, threshold):
    # Calculate variances using VarianceThreshold
    selector = VarianceThreshold(threshold=0.0)  # Doesn't filter any feature initially
    selector.fit(positional_weights)
    variances = selector.variances_

    # Get variances and rank indices
    ranked_indices = np.argsort(variances)[::-1] # from highest to lowest variance -- as high variance are preferred
    ranked_variances = variances[ranked_indices]

    n = int(len(variances) * threshold)
    capped_indices = ranked_indices[:n]
    print(f"Getting top {n} indices")
    return capped_indices



# ================= get simple variance indices =================
parent_dir = "main-exp-result/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9"
positional_weights, mean_pw, std_pw = get_positional_weights(parent_dir)
threshold = 0.06
capped_indices = get_simple_variance_indices(positional_weights, threshold)
# np.save(os.path.join(parent_dir, f"variance_indices_perc{int(threshold*100)}.npy"), capped_indices)
# =============================================================