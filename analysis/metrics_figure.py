import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import math
from matplotlib.ticker import MaxNLocator
import itertools
import matplotlib.gridspec as gridspec
from sympy.abc import alpha


def get_auroc(trial_result_folder:str, test_data):
    if test_data == "internal_test":
        pkl_filename = os.path.join(trial_result_folder, "sum_of_scores.pkl")
    elif test_data == "external_test":
        pkl_filename = os.path.join(trial_result_folder, "unseen_sum_of_scores.pkl")

    with open(pkl_filename, "rb") as file:
        test_metrics = pickle.load(file)

    aurocs = test_metrics['auroc']

    return aurocs





# ================ Truncation results plot ================
# aurocs_set1 = get_auroc("study1-truncate-result/trial-result-VLP200-ridge-onehotprotein-seqlen200-optimfold9", "internal_test")
# aurocs_set2 = get_auroc("study1-truncate-result/trial-result-VLP200-ridge-onehotprotein-seqlen400-optimfold9", "internal_test")
# aurocs_set3 = get_auroc("study1-truncate-result/trial-result-VLP200-ridge-onehotprotein-seqlen600-optimfold9", "internal_test")
# aurocs_set4 = get_auroc("study1-truncate-result/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9", "internal_test")


# data_left = [aurocs_set1, aurocs_set2, aurocs_set3, aurocs_set4]
#
# # Calculate means and standard deviations
# means_left = [np.mean(data) for data in data_left]
# stds_left = [np.std(data) for data in data_left]
#
# # Set up the positions
# x_left = np.arange(len(data_left))
#
#
# # Set up the matplotlib figure
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # Plot bar charts with error bars
# ax.bar(x_left, means_left, yerr=stds_left, capsize=5, label='Cross-validation set', color = sns.color_palette("colorblind6"))
#
# # Set labels, ticks, and title
# ax.set_xticks(x_left)
# ax.set_xticklabels([r'$L=200$', r'$L=400$', r'$L=600$', r'$L=1426$'])
# ax.set_ylabel('AUROC', fontsize=12)
# ax.tick_params(axis='y', labelsize=12)
# ax.tick_params(axis='x', labelsize=12)
# ax.set_ylim(0.7, 0.95)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
#
# # Add group labels
# ax.text(np.mean(x_left), 0.66, 'on cross-validation set', ha='center', fontsize=12)
#
# # Adjust layout and save
# plt.margins(x=0.05)
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.savefig('ridge-onehot-truncation_results.png', dpi=300)
# plt.show()

# ================ Truncation results box plot ================
# data = [aurocs_set1, aurocs_set2, aurocs_set3, aurocs_set4]
#
# # Set up the matplotlib figure
# fig, ax = plt.subplots(figsize=(6, 5))
#
#
# # Set the face color palette
# face_colors = sns.color_palette("colorblind6")
#
# # Create the boxplot
# fig, ax = plt.subplots(figsize=(8,6))
# sns.boxplot(data=data, palette=face_colors, ax=ax)
#
# # Set labels, ticks, and title
# x_left = np.arange(4)  # Set x positions for the labels
# ax.set_xticks(x_left)
# ax.set_xticklabels([r'$L=200$', r'$L=400$', r'$L=600$', r'$L=1426$'])
# ax.set_ylabel('AUROC', fontsize=14)
# ax.tick_params(axis='y', labelsize=14)
# ax.tick_params(axis='x', labelsize=14)
# ax.set_ylim(0.5, 1.0)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
#
# # Adjust layout and save
# plt.margins(x=0.05)
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.savefig('ridge-onehot-truncation_results.png', dpi=300)
# plt.show()
# ==========================================================



# ================ Selected positions results plot  ================
# aurocs_trunc200= get_auroc("study1-truncate-result/trial-result-VLP200-ridge-onehotprotein-seqlen200-optimfold9", "internal_test")
# aurocs_wei5perc = get_auroc("study2-top-position/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9-perc5", "internal_test")
# aurocs_lap22perc = get_auroc("study2-top-position/trial-result-VLP200-laplacian-ridge-onehotprotein-seqlen1426-optimfold9-perc22", "internal_test")
# aurocs_var5perc = get_auroc("study2-top-position/trial-result-VLP200-variance-ridge-onehotprotein-seqlen1426-optimfold9-perc5", "internal_test")
#
# data_left = [aurocs_trunc200, aurocs_wei5perc, aurocs_lap22perc, aurocs_var5perc]
#
# # Calculate means and standard deviations
# means_left = [np.mean(data) for data in data_left]
# stds_left = [np.std(data) for data in data_left]
#
#
# # Set up the positions
# x_left = np.arange(len(data_left))
#
#
# # Set up the matplotlib figure
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # Plot bar charts with error bars
# ax.bar(x_left, means_left, yerr=stds_left, capsize=5, label='Cross-validation set', color = sns.color_palette("Set2"))
#
# print(means_left)
#
# # Set labels, ticks, and title
# ax.set_xticks(np.concatenate([x_left]))
# ax.set_xticklabels(
#     [
#         'Truncation\n(200, 14%)',
#         'Weights ranking\n(71, 5%)',
#         'Laplacian score\n(313, 22%)',
#         'Variance ranking\n(71, 5%)'
#     ],
# )
# ax.set_ylabel('AUROC', fontsize=14)
# ax.tick_params(axis='y', labelsize=14)
# ax.tick_params(axis='x', labelsize=14)
# ax.set_ylim(0.7, 0.95)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
#
# # Adjust layout and save
# plt.margins(x=0.05)
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.savefig('selected-positions_results.png', dpi=300)
# plt.show()





# ================ Overall Results plot ================
# aurocs_set1 = get_auroc("study1-truncate-result/trial-result-VLP200-ridge-onehotprotein-seqlen200-optimfold9", "internal_test")
# aurocs_set2 = get_auroc("study2-top-position/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9-perc12", "internal_test")
# aurocs_set3 = get_auroc("study2-top-position/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9-perc12", "internal_test")
#
# aurocs_set4 = get_auroc("study1-truncate-result/trial-result-VLP200-ridge-onehotprotein-seqlen200-optimfold9", "external_test")
# aurocs_set5 = get_auroc("study2-top-position/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9-perc12", "external_test")
# aurocs_set6 = get_auroc("study2-top-position/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9-perc12", "external_test")
#
# data_left = [aurocs_set1, aurocs_set2]
# data_right = [aurocs_set4, aurocs_set5]
#
# # Set up the matplotlib figure
# # Create figure and axes
# fig, ax = plt.subplots(figsize=(8, 5))
#
#
# # Set positions for the left and right grou12\%ps of boxplots
# x_left = np.arange(len(data_left)) * 0.4 # Set space within groups
# x_right = np.arange(len(data_right)) * 0.4  + len(data_left) -1  # Set space between groups
#
#
# colours = [sns.color_palette("colorblind6")[0], sns.color_palette("Set2")[2]]
#
# # Plot left group of boxplots
# bp_left = ax.boxplot(data_left, positions=x_left, patch_artist=True, )
# for patch, color in zip(bp_left['boxes'], colours):
#     patch.set_facecolor(color)
#
# # Plot right group of boxplots
# bp_right = ax.boxplot(data_right, positions=x_right, patch_artist=True,)
# for patch, color in zip(bp_right['boxes'], colours):
#     patch.set_facecolor(color)
#
# # Set labels, ticks, and title
# ax.set_xticks(np.concatenate([x_left, x_right]))
# ax.set_xticklabels([r'$Trunc.L=200$', r'$Select \ 12\%$', r'$Trunc.L=200$', r'$Select \ 12\%$'])
#
#
# plt.margins(x=-0.05)
# ax.set_ylabel('AUROC', fontsize=12)
# ax.tick_params(axis='y', labelsize=12)
# ax.tick_params(axis='x', labelsize=12)
#
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
#
# ax.set_ylim(0.5, 1.0)
# ax.tick_params(axis='x', length=0)
#
# # ax.set_xlabel('by Protein Length ', fontsize=14, labelpad=30)
# # ax.set_title('Truncation results')
#
# # Show the plot
# plt.tight_layout()
#
# ax.text(np.mean(x_left), 0.43, 'on cross-validation set', ha='center', fontsize=12)
# ax.text(np.mean(x_right), 0.43, 'on hold-out set', ha='center', fontsize=12)
# plt.subplots_adjust(bottom=0.15)
#
# plt.savefig('positional-influence-result-overall.png', dpi=300)
# plt.show()
#


# ================ scatter plot for each method ================

def scatterplot(x, means, stds, fmt, color, filename):
    #  Change x to scaled x
    x_labels = x
    # use log scale for x-axis
    # x = np.log(x)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # plot the scatter plot with error bars, and set the color of scatter to be red,
    ax.errorbar(x, means, yerr=stds, fmt=fmt, capsize=5,
                label='Cross-validation set', color=color, ecolor="grey")

    # add ylimit
    ax.set_ylim(0.55, 1.0)

    # change the x-axis to the original label
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}%' for i in x_labels], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.xticks(ticks=[1, 5, 10, 20, 30, 40, 50], labels=["1%", "5%", "10%", "20%", "30%", "40%", "50%"], rotation=0)



    # Add grey grid lines
    plt.grid(color='grey', linestyle='-', linewidth=1.5, alpha=0.15)

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)





    # Highlight: identify the point with the highest mean value
    max_idx = np.argmax(means)  # Index of the max mean
    max_x = x[max_idx]  # Corresponding x value
    max_y = means[max_idx]  # Maximum mean value

    # Highlight the max mean point
    ax.scatter(max_x, max_y, color="black", marker="*", s=250, zorder=3, label="Max AUROC", alpha=0.6)  # Large red point

    # Add grid lines marking the max point
    ax.hlines(y=max_y, xmin=ax.spines['left'].get_position()[0], xmax=max_x, color="grey", linestyle='--', linewidth=1, alpha=0.6)
    ax.vlines(x=max_x, ymin=ax.get_ylim()[0], ymax=max_y, color="grey", linestyle='--', linewidth=1, alpha=0.6)

    # Annotate the max point
    # === For truncation plot ===
    ax.annotate(f'({max_x}%, {max_y:.2f})',
                xy=(max_x, max_y), xytext=(max_x + 0.7, max_y - 0.1),
                fontsize=12, color="black", fontweight='bold', alpha=0.8,
                arrowprops=dict(facecolor=sns.color_palette()[3], arrowstyle="-", lw=1.5, alpha=0.6))
    # === For by weights plot ===
    # ax.annotate(f'({max_x}%, {max_y:.2f})',
    #             xy=(max_x, max_y), xytext=(max_x + 1.0, max_y + 0.06),
    #             fontsize=12, color="black", fontweight='bold', alpha=0.8,
    #             arrowprops=dict(facecolor=sns.color_palette()[3], arrowstyle="-", lw=1.5, alpha=0.6))

    # === For Laplacian plot ===
    # ax.annotate(f'({max_x}%, {max_y:.2f})',
    #             xy=(max_x, max_y), xytext=(max_x - 2, max_y + 0.08),
    #             fontsize=12, color="black", fontweight='bold', alpha=0.8,
    #             arrowprops=dict(facecolor=sns.color_palette()[3], arrowstyle="-", lw=1.5, alpha=0.6))

    # === For variance plot ===
    # ax.annotate(f'({max_x}%, {max_y:.2f})',
    #             xy=(max_x, max_y), xytext=(max_x - 2, max_y - 0.15),
    #             fontsize=12, color="black", fontweight='bold', alpha=0.8,
    #             arrowprops=dict(facecolor=sns.color_palette()[3], arrowstyle="-", lw=1.5, alpha=0.6))



    # set y label font size
    ax.set_ylabel('AUROC', fontsize=14)
    ax.set_xlabel('Percentage of positions selected(%)', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()




#  ================ Selected positions results plot: Truncation plot ================
folders = []
for folder in os.listdir("study1-truncate-result/"):
    if folder.startswith("trial-result-VLP200-ridge-onehotprotein-seqlen"):
        if any(substring in folder for substring in ("seqlen50", "seqlen100", "seqlen114", "seqlen185",
                                                     "seqlen199", "seqlen143", "seqlen200", "seqlen400",
                                                     "seqlen600",  "seqlen1426")):
        # if "seqlen1426" in folder:
            continue
        else:
            folders.append(folder)
# sort the folder names by the number after "seqlen"
folders = sorted(folders, key=lambda x: int(x.split("seqlen")[-1].replace("-optimfold9", "")))
# get the number after "seqlen" in the folder names
x = [int(folder.split("seqlen")[-1].replace("-optimfold9", "")) for folder in folders]
print(x)

# convert x to percentage
x = [round(i * 100 / 1426) for i in x]
# print(x)
folders = [os.path.join("study1-truncate-result/", folder) for folder in folders]

# get the auroc values for each folder
aurocs = [get_auroc(folder, "internal_test") for folder in folders]


# get mean and std
means = [np.mean(auroc) for auroc in aurocs]
stds = [np.std(auroc) for auroc in aurocs]
print(means)
scatterplot(x, means, stds,"s", sns.color_palette("Set2")[3], filename="truncation-result.pdf")


#  ================ Selected positions results plot: Weights plot ================
# Under the folder "study2-top-position/" Find folder start with the name "trial-result-VLP200-variance-ridge-onehotprotein-seqlen1426-optimfold9"
# folders = []
# for folder in os.listdir("study2-top-position/"):
#     if folder.startswith("trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9"):
#         if folder.endswith(("perc2", "perc3", "perc5", "perc7", "perc9", "perc11", "perc12", "perc13", "perc14", "perc16", "perc45", "perc50")):
#             continue
#         else:
#             folders.append(folder)
# print(len(folders))
# # sort the folder names by the last number in the folder name
# folders = sorted(folders, key=lambda x: int(x.split("-")[-1].replace("perc", "")))
# # get the last number in the folder name in the folder names
# x = [int(folder.split("-")[-1].replace("perc", "")) for folder in folders]
#
# print(x)
#
# folders = [os.path.join("study2-top-position/", folder) for folder in folders]
#
# # get the auroc values for each folder
# aurocs = [get_auroc(folder, "internal_test") for folder in folders]
# print(x)
# print( aurocs)
# # get mean and std
# means = [np.mean(auroc) for auroc in aurocs]
# stds = [np.std(auroc) for auroc in aurocs]
#
# scatterplot(x, means, stds,"o", sns.color_palette("Set2")[0], filename="weightsranking-result.pdf")
#


#  ================ Selected positions results plot: Laplacian plot ================
# Under the folder "study2-top-position/" Find folder start with the name "trial-result-VLP200-variance-ridge-onehotprotein-seqlen1426-optimfold9"
# folders = []
# for folder in os.listdir("study2-top-position/"):
#     if folder.startswith("trial-result-VLP200-laplacian-ridge-onehotprotein-seqlen1426-optimfold9"):
#         if folder.endswith(("perc12", "perc22", "perc23", "perc24", "perc26", "perc36","perc38",
#                             "perc42", "perc44", "perc50",)):
#             continue
#         else:
#             folders.append(folder)
#
# # sort the folder names by the last number in the folder name
# folders = sorted(folders, key=lambda x: int(x.split("-")[-1].replace("perc", "")))
# # get the last number in the folder name in the folder names
# x = [int(folder.split("-")[-1].replace("perc", "")) for folder in folders]
# print(x)
#
# folders = [os.path.join("study2-top-position/", folder) for folder in folders]
#
# # get the auroc values for each folder
# aurocs = [get_auroc(folder, "internal_test") for folder in folders]
# # get mean and std
# means = [np.mean(auroc) for auroc in aurocs]
# stds = [np.std(auroc) for auroc in aurocs]
#
# print(means)
#
# scatterplot(x, means, stds, "D", sns.color_palette("Set2")[2], filename="laplacian-result.pdf")



# #  ================ Selected positions results plot: Variance plot ================
# Under the folder "study2-top-position/" Find folder start with the name "trial-result-VLP200-variance-ridge-onehotprotein-seqlen1426-optimfold9"
# folders = []
# for folder in os.listdir("study2-top-position/"):
#     if folder.startswith("trial-result-VLP200-variance-ridge-onehotprotein-seqlen1426-optimfold9"):
#         if folder.endswith(("perc2", "perc3", "perc4", "perc6", "perc7", "perc8","perc9",
#                             "perc12", "perc13", "perc14", "perc16",  "perc18", "perc23", "perc25", "perc28",
#                             "perc50", )):
#             continue
#         else:
#             folders.append(folder)
#
# # sort the folder names by the last number in the folder name
# folders = sorted(folders, key=lambda x: int(x.split("-")[-1].replace("perc", "")))
# # get the last number in the folder name in the folder names
# x = [int(folder.split("-")[-1].replace("perc", "")) for folder in folders]
# print(x)
# folders = [os.path.join("study2-top-position/", folder) for folder in folders]
#
# # get the auroc values for each folder
# aurocs = [get_auroc(folder, "internal_test") for folder in folders]
#
# # get mean and std
# means = [np.mean(auroc) for auroc in aurocs]
# stds = [np.std(auroc) for auroc in aurocs]
# print(means)
# scatterplot(x, means, stds, "^", sns.color_palette("Set2")[1], filename="variance-result.pdf")
#
#
