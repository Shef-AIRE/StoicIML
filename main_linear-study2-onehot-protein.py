import argparse

from attr.validators import max_len

from configs.configs import get_cfg_defaults
from time import time
import pandas as pd
from functools import partial
from collections import Counter
import numpy as np
import os
import pickle
import yaml

from pipeline._utils import mkdir, result_holder
from pipeline._split import KFoldSplit
from pipeline._linear_beta_study2_onehot_protein import LinearTrainer, test


def arg_parse():
    parser = argparse.ArgumentParser(description='Protein formation to virus-like-particles Prediction')
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--model", type=str, required=True, help="model type, e.g. 'logit', 'svm', 'ridge'")
    parser.add_argument("--positional_index", type=str, required=True, help="file to positional index")
    args = parser.parse_args()
    return args

def  main():
    s = time()

    # ---- setup config ----
    cfg = get_cfg_defaults()
    args = arg_parse()
    cfg.merge_from_file(args.config)

    # ---- setup dataset ----
    datapath = f"{cfg.SEQ.INPUT_PATH}"
    df = pd.read_csv(datapath, usecols=['Protein sequence', 'Stoichiometry'])
    df = df[df['Stoichiometry'].isin([60, 180])]  # prediction of stoichiometry = 60 or 180
    df['Stoichiometry'].replace({60: 0, 180: 1}, inplace=True) # stoichiometry= 60->0: 51 samples, stoichiometry=180->1: 49 samples


    # ---- setup unseen dataset ----
    unseen_datapath = f'./datasets/curated/VLP_unseen.csv'
    df_unseen = pd.read_csv(unseen_datapath, usecols=['Protein sequence', 'Stoichiometry'])
    df_unseen = df_unseen[df_unseen['Stoichiometry'].isin([60, 180])]
    df_unseen['Stoichiometry'].replace({60: 0, 180: 1}, inplace=True) # stoichiometry= 60->0: 97 samples, stoichiometry=180->1: 60 samples



    # ---- max length of protein sequence ----
    # max_length = max(df['Protein sequence'].apply(len))
    # print(f"Max length of protein sequence: {max_length}")
    # print()

    # ---- positional index ----
    positional_index = np.load(args.positional_index)
    perc = args.positional_index.split('_')[-1]
    perc = perc.split('.')[0]

    # ---- setup output directory ----
    export_dir = f"{cfg.RESULT.OUTPUT_DIR}-{args.model}-onehotprotein-seqlen{cfg.SEQ.MAX_LEN}-optimfold{cfg.CV.OPTIM_FOLD}-{perc}"
    mkdir(export_dir)

    # ---- set Model SEED ----
    SEED = cfg.MODEL.SEED
    # set_seed(SEED)
    # random.seed(SEED)
    # np.random.seed(SEED)


    # ---- setup result dict ----
    sum_of_scores, misc, hyperparameters = result_holder()
    unseen_sum_of_scores, unseen_misc, _ = result_holder()


    ### LOOP1: RANDOM REPEATS
    for seed in range(cfg.CV.REPEATS):
        # print(f"CV seed: {seed}")
        TrainTestSplit = KFoldSplit(df, cfg.CV.N_FOLD, seed, 'Stoichiometry')


        ### LOOP2: TRAIN-TEST SPLIT INTO N_FOLD

        for fold, (idx_train, idx_test) in enumerate(TrainTestSplit.split()):
            print(f"CV seed: {seed}, Test fold: {fold}")

            df_train, df_test = df.iloc[idx_train], df.iloc[idx_test]

            if not (df_test.index.tolist() == df.index[idx_test]).all():
                raise ValueError(
                    f"Mismatch in indices: df_test.index is {df_test.index.tolist()}, but df.index[idx_test] is {df.index[idx_test]}")

            # Sanity check: Stratification is applied correctly
            # print(f"Train: {df_train['Stoichiometry'].value_counts()}, Test: {df_test['Stoichiometry'].value_counts()}")




            ### LOOP3: TRAIN-VAL SPLIT INTO OPTIM_FOLD
            linear_trainer = LinearTrainer()

            objective_with_params = partial(
                linear_trainer.objective,
                # model construction parameters
                linear_classifier=args.model,
                random_state=cfg.MODEL.SEED,

                # data resampling parameters
                df=df_train,
                n_folds=cfg.CV.OPTIM_FOLD,
                seed=seed,
                stratify_column_name='Stoichiometry',
                max_length=cfg.SEQ.MAX_LEN,
                select_index=positional_index,
                reduce_variance=False
            )

            best_framework, metadata = linear_trainer.optimise(
                objective_with_params,
                random_state=cfg.MODEL.SEED,
                n_startup_trials = 100,  # for global search
                n_trials=50, # Bayesian for local search

                export_dir= export_dir,
            )

            hyperparameters.update({
                f"seed{seed}_fold{fold}": {
                    "best_framework": best_framework,
                    "metadata": metadata
                }
            })

            test(framework=best_framework,

                 df_train=df_train,  # data parameters
                 df_test=df_test,

                 sum_of_scores=sum_of_scores,
                 misc=misc,
                 max_length=cfg.SEQ.MAX_LEN,
                 select_index=positional_index,
                 )

            # ----- test on unseen dataset -----
            test(framework=best_framework,

                 df_train=df_train,  # data parameters
                 df_test=df_unseen,

                 sum_of_scores=unseen_sum_of_scores,
                 misc=unseen_misc,
                 max_length=cfg.SEQ.MAX_LEN,
                 select_index=positional_index,
                 )


    # ---- append running time to sum_of_scores ----
    e = time()
    total_running_time = round(e - s, 2)
    print(f"Total running time: {total_running_time}s")
    sum_of_scores["running_time"] = total_running_time
    unseen_sum_of_scores["running_time"] = total_running_time


    # ----- Export results -----
    # print(sum_of_scores)
    print(len(misc["model_weights"]), len(misc["train_index"]), len(misc["test_index"]))

    print("accuracy mean", np.mean(sum_of_scores["accuracy"]), "std", np.std(sum_of_scores["accuracy"]))
    print("auroc mean", np.mean(sum_of_scores["auroc"]), "std", np.std(sum_of_scores["auroc"]))
    print("recall mean", np.mean(sum_of_scores["recall"]), "std", np.std(sum_of_scores["recall"]))
    print("specificity mean", np.mean(sum_of_scores["specificity"]), "std", np.std(sum_of_scores["specificity"]))
    print("precision mean", np.mean(sum_of_scores["precision"]), "std", np.std(sum_of_scores["precision"]))
    print("npv mean", np.mean(sum_of_scores["npv"]), "std", np.std(sum_of_scores["npv"]))


    with open(os.path.join(export_dir, "sum_of_scores.pkl"), "wb") as file:
        pickle.dump(sum_of_scores, file)

    with open(os.path.join(export_dir, "misc.pkl"), "wb") as file:
        pickle.dump(misc, file)

    with open(os.path.join(export_dir, "best_frameworks.yml"), "w") as f:
        yaml.dump(hyperparameters, f)

    # ----- Export results on unseen dataset -----
    print("accuracy mean", np.mean(unseen_sum_of_scores["accuracy"]), "std", np.std(unseen_sum_of_scores["accuracy"]))
    print("auroc mean", np.mean(unseen_sum_of_scores["auroc"]), "std", np.std(unseen_sum_of_scores["auroc"]))

    with open(os.path.join(export_dir, "unseen_sum_of_scores.pkl"), "wb") as file:
        pickle.dump(unseen_sum_of_scores, file)

    with open(os.path.join(export_dir, "unseen_misc.pkl"), "wb") as file:
        pickle.dump(unseen_misc, file)






if __name__ == '__main__':
    # s = time()
    result = main()
    # e = time()
    # print(f"Total running time: {round(e-s,2)}s")