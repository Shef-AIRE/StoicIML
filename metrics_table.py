import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec



def calculate_test_mean_std(trial_result_folder:str, test_data) -> tuple:
    if test_data == "internal_test":
        pkl_filename = os.path.join(trial_result_folder, "sum_of_scores.pkl")
    elif test_data == "external_test":
        pkl_filename = os.path.join(trial_result_folder, "unseen_sum_of_scores.pkl")

    with open(pkl_filename, "rb") as file:
        test_metrics = pickle.load(file)

    running_time = test_metrics["running_time"]
    metric_keys = ['accuracy', 'auroc', 'recall', 'specificity', 'precision', 'npv']
    means = {metric: np.mean(test_metrics[metric]) for metric in metric_keys}
    stds = {metric: np.std(test_metrics[metric]) for metric in metric_keys}

    return running_time, means, stds
    # return means, stds


# ---- Single result example -----
# means, stds = calculate_test_mean_std(
#     "./main-exp-result/trial-result-VLP100-logit-seqlen736-optimfold9",
#     test_data="internal_test"
# )
# for (key, mean), (key2, std) in zip(means.items(), stds.items()):
#     print(f"{key}: {mean:.2f} ± {std:.2f}")


# ----- Sanity experiment, Main experiment -----

# print("\n")
#
# df_list = []
# for train_data in ["VLP200"]: # ["VLP100", "VLP200"]:
#     # if train_data == "VLP100":
#     #     seq_length = "seqlen736"
#     # elif train_data == "VLP200":
#     seq_length = "seqlen1426"
#     for test_data in ["internal_test", "external_test"]:
#         for model in ["ridge", "svm"]:# ["logit", "ridge", "svm"]:
#             print(f"Train \"{train_data}\", Test \"{test_data},\" Model \"{model}\"")
#             trial_result_folder = f"./main-exp-result/trial-result-{train_data}-{model}-{seq_length}-optimfold9"
#             running_time, means, stds = calculate_test_mean_std(trial_result_folder, test_data)
#
#             result_dict = {
#                 "train_data": train_data,
#                 "test_data": test_data,
#                 "model": model,
#                 "running_time": running_time,
#                 "accuracy": f"{means['accuracy']:.2f} ± {stds['accuracy']:.2f}",
#                 "auroc": f"{means['auroc']:.2f} ± {stds['auroc']:.2f}",
#                 "recall": f"{means['recall']:.2f} ± {stds['recall']:.2f}",
#                 "specificity": f"{means['specificity']:.2f} ± {stds['specificity']:.2f}",
#                 "precision": f"{means['precision']:.2f} ± {stds['precision']:.2f}",
#                 "npv": f"{means['npv']:.2f} ± {stds['npv']:.2f}"
#             }
#             df_list.append(result_dict)
#
# result_df = pd.DataFrame(df_list)
# result_df.to_csv("./main-exp-result/Compare_VLP100_VLP200.csv", index=False)
# print(result_df.columns)


# ----- Study 1: Truncation by sequence length -----
# print("\n")
#
# study1_df_list = []
# for encoding in ["intlabelprotein"]: #, "onehotprotein"]:
#     for test_data in ["internal_test", "external_test"]:
#         for model in  ["ridge", "svm", "logit"]:
#             for seq_length in ["seqlen200", "seqlen400", "seqlen600", "seqlen1426"]:
#                 print(f"Model \"{model}\", Sequence length \"{seq_length}\"")
#                 trial_result_folder = f"./study1-truncate-result/trial-result-VLP200-{model}-{encoding}-{seq_length}-optimfold9"
#                 running_time, means, stds = calculate_test_mean_std(trial_result_folder, test_data)
#
#                 result_dict = {
#                     "encoding": encoding,
#                     "test_data": test_data,
#                     "model": model,
#                     "seq_length": seq_length,
#                     "running_time": running_time,
#                     "accuracy": f"{means['accuracy']:.2f} ± {stds['accuracy']:.2f}",
#                     "auroc": f"{means['auroc']:.2f} ± {stds['auroc']:.2f}",
#                     "recall": f"{means['recall']:.2f} ± {stds['recall']:.2f}",
#                     "specificity": f"{means['specificity']:.2f} ± {stds['specificity']:.2f}",
#                     "precision": f"{means['precision']:.2f} ± {stds['precision']:.2f}",
#                     "npv": f"{means['npv']:.2f} ± {stds['npv']:.2f}"
#                 }
#                 study1_df_list.append(result_dict)
#
# study1_result_df = pd.DataFrame(study1_df_list)
# study1_result_df.to_csv("./study1-truncate-result/Study1_truncate.csv", index=False)
# print(study1_result_df.columns)


# ----- Sanity experiment 2: Find the best optimisation metric
# print("\n")
#
# sanity2_df_list = []
# for model in ["logit", "ridge", "svm"]:
#     for optim_metric in ["accuracyOptim", "aurocOptim", "mccOptim"]:
#         print(f"Optim metric \'{optim_metric}\' , Model \"{model}\"")
#         trial_result_folder = f"./sanity2-score-to-optim/trial-result-VLP200-{optim_metric}-{model}-seqlen1426-optimfold9"
#         means, stds = calculate_test_mean_std(trial_result_folder, "internal_test")
#
#         result_dict = {
#             "model": model,
#             "optim_metric": optim_metric,
#             "accuracy": f"{means['accuracy']:.2f} ± {stds['accuracy']:.2f}",
#             "auroc": f"{means['auroc']:.2f} ± {stds['auroc']:.2f}",
#             "recall": f"{means['recall']:.2f} ± {stds['recall']:.2f}",
#             "specificity": f"{means['specificity']:.2f} ± {stds['specificity']:.2f}",
#             "precision": f"{means['precision']:.2f} ± {stds['precision']:.2f}",
#             "npv": f"{means['npv']:.2f} ± {stds['npv']:.2f}"
#         }
#         sanity2_df_list.append(result_dict)
#
# sanity2_result_df = pd.DataFrame(sanity2_df_list)
# sanity2_result_df.to_csv(f"./sanity2-score-to-optim/Sanity2_score_to_optim.csv", index=False)
# print(sanity2_result_df.columns)



# ----- Study 2: Find top positions -----
# print("\n")
#
# study2_df_list = []
# for test_data in ["internal_test", "external_test"]:
#     # for perc in ["perc5", "perc10", "perc20", "perc22",  "perc23", "perc24",  "perc26", "perc28", "perc30", "perc40", "perc50"]:
#     for perc in ["perc5", "perc10", "perc12", "perc13", "perc14", "perc15", "perc16", "perc18", "perc20", "perc30", "perc40", "perc50"]:
#         trial_result_folder = f"./study2-top-position/trial-result-VLP200-variance-ridge-onehotprotein-seqlen1426-optimfold9-{perc}"
#         running_time, means, stds = calculate_test_mean_std(trial_result_folder, test_data)
#
#         result_dict = {
#             "test_data": test_data,
#             "perc": perc,
#             "running_time": running_time,
#             "accuracy": f"{means['accuracy']:.2f} ± {stds['accuracy']:.2f}",
#             "auroc": f"{means['auroc']:.2f} ± {stds['auroc']:.2f}",
#             "recall": f"{means['recall']:.2f} ± {stds['recall']:.2f}",
#             "specificity": f"{means['specificity']:.2f} ± {stds['specificity']:.2f}",
#             "precision": f"{means['precision']:.2f} ± {stds['precision']:.2f}",
#             "npv": f"{means['npv']:.2f} ± {stds['npv']:.2f}"
#         }
#         study2_df_list.append(result_dict)
#
# study2_result_df = pd.DataFrame(study2_df_list)
# study2_result_df.to_csv(f"./study2-top-position/Study2_top_position.csv", index=False, sep=",", encoding="utf-8")
# print(study2_result_df.columns)


# ----- Study 3: Amino acid cluster significance -----
# print("\n")
#
# study3_df_list = []
# for test_data in ["internal_test", "external_test"]:
#     for encoding in ["onehot-protein"]:
#         for model in ["logit"]:
#             trial_result_folder = f"./study3-cluster/trial-result-VLP200-{encoding}-{model}-seqlen1426-optimfold9"
#             running_time, means, stds = calculate_test_mean_std(trial_result_folder, test_data)
#
#             result_dict = {
#                 "test_data": test_data,
#                 "encoding": encoding,
#                 "model": model,
#                 "running_time": running_time,
#                 "accuracy": f"{means['accuracy']:.2f} ± {stds['accuracy']:.2f}",
#                 "auroc": f"{means['auroc']:.2f} ± {stds['auroc']:.2f}",
#                 "recall": f"{means['recall']:.2f} ± {stds['recall']:.2f}",
#                 "specificity": f"{means['specificity']:.2f} ± {stds['specificity']:.2f}",
#                 "precision": f"{means['precision']:.2f} ± {stds['precision']:.2f}",
#                 "npv": f"{means['npv']:.2f} ± {stds['npv']:.2f}"
#             }
#             study3_df_list.append(result_dict)
#
#     study3_result_df = pd.DataFrame(study3_df_list)
#     study3_result_df.to_csv(f"./study3-cluster/Study3_cluster.csv", index=False)
#     print(study3_result_df.columns)





