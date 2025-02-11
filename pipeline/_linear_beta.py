import yaml
import os
import shutil
import numpy as np
import pickle

import optuna
from optuna.samplers import TPESampler
from optuna import Study, Trial


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix
from imblearn.metrics import specificity_score

from pipeline._split import RandomSplit, KFoldSplit
from pipeline._dataloader import to_embedding


class LinearTrainer:
    def __init__(self):
        self.framework_store = {}

    def objective(self, trial,
                # model construction parameters
                  linear_classifier=None,
                  penalty=None,
                  max_iter=1000,
                  random_state=None,
                  sep=".",
                  prefix=None,

                # data resampling parameters
                  df=None,
                  n_folds=None,
                  seed=None,
                  stratify_column_name=None,
                  max_length=None,
                  reduce_variance=None
                  ):



        # ---- construct a trial model with possible hyperparameters ----
        framework = {}


        # ------ set max_iter and random_state parameters ------
        framework["params"] = {}
        framework["params"]["max_iter"] = max_iter
        framework["params"]["random_state"] = random_state


        # ----- select linear classifier -----
        if linear_classifier is None:
            linear_classifier = trial.suggest_categorical(
                name="name",
                choices=["logit", "svm", "ridge"]
            )
            prefix = sep.join(filter(None, (prefix, linear_classifier)))
            framework["name"] = linear_classifier



        # ----- select regularization parameter C (log scale) ----
        framework["params"]["C"] = 2 ** trial.suggest_int(
            name=sep.join(filter(None, (prefix, "C_log2"))),
            low=-5, high=15
        )  # regularisation parameter


        # ------ select class weight ------
        framework["params"]["class_weight"] = trial.suggest_categorical(
            name=sep.join(filter(None, (prefix, "class_weight"))),
            choices=("balanced", None),
        )



        # ------ logistic regression -------
        if linear_classifier == "logit":
            framework["name"] = linear_classifier
            if penalty is None:
                penalty = trial.suggest_categorical(
                    name=sep.join(filter(None, (prefix, "penalty"))),
                    choices=("l1", "l2", "elasticnet"),
                )
            framework["params"]["penalty"] = penalty

            if penalty == "elasticnet":
                l1_ratio = trial.suggest_float(
                    name=sep.join(filter(None, (prefix, "l1_ratio"))),
                    low=0.0,
                    high=1.0,
                    step=0.05,
                )
                framework["params"]["l1_ratio"] = l1_ratio

            if penalty == "elasticnet":
                solver = "saga"  # saga is the only solver that supports Elastic Net.
            else:
                solver = "liblinear" # Small to medium-sized datasets, L1/L2 regularisation.
            framework["params"]["solver"] = solver

            classifier_obj = LogisticRegression(**framework["params"])

        # ------ linear SVM ------
        if linear_classifier == "svm":
            framework["name"] = linear_classifier
            if penalty is None:
                penalty = trial.suggest_categorical(
                    name=sep.join(filter(None, (prefix, "penalty"))),
                    choices=("l1", "l2"),
                )
            framework["params"]["penalty"] = penalty

            if penalty == "l1":
                loss = "squared_hinge"
                dual = False
            if penalty == "l2":
                loss = trial.suggest_categorical(
                    name=sep.join(filter(None, (prefix, "loss"))),
                    choices=("hinge", "squared_hinge"),
                )
                dual = True
            framework["params"]["dual"] = dual
            framework["params"]["loss"] = loss

            classifier_obj = LinearSVC(**framework["params"])


        # ----- ridge regression -----
        if linear_classifier == "ridge":
            framework["name"] = linear_classifier

            framework["params"]["alpha"] = 1 / (2 * framework["params"].pop("C"))
            classifier_obj = RidgeClassifier(**framework["params"])

        print(classifier_obj)

        # ---- store the framework ----
        # print(framework)
        self.framework_store[trial.number] = framework


        # ---- train and validation data resampling ----
        sum_of_scores = []
        TrainValSplit = KFoldSplit(df, n_folds, seed, stratify_column_name)


        for fold, (idx_train, idx_val) in enumerate(TrainValSplit.split()):
            print(f"optim fold: {fold}")
            # ---- setup dataset ----
            df_train, df_val = df.iloc[idx_train], df.iloc[idx_val]

            # Sanity check: Stratification is applied correctly
            # print(f"Train: {df_train['Stoichiometry'].value_counts()}, Val: {df_val['Stoichiometry'].value_counts()}")

            train_index, train_x, train_y = to_embedding(df_train, max_length)
            val_index, val_x, val_y = to_embedding(df_val, max_length)


            if not (val_index == df.index[idx_val]).all():
                raise ValueError(
                    f"Mismatch in indices: val_index is {val_index}, but df.index[idx_val] is {df.index[idx_val]}")


            # ---- train and validation data ----
            clf = classifier_obj.fit(train_x, train_y)
            y_pred = clf.predict(val_x)
            accuracy = accuracy_score(val_y, y_pred)
            sum_of_scores.append(accuracy)


        score = np.mean(sum_of_scores)

        # ---- Optionally reduce variance in the score ----
        if reduce_variance:
            score = score - np.std(sum_of_scores, ddof=1)
        # ---- return the mean of the scores ----
        return score






    def optimise(self,
                 objective, # objective function
                 random_state=None,
                 n_startup_trials=None,
                 n_trials=None,

                 export_dir=None
                 ):

        # ---- Bayesian Optimisation: TPE to optimise objective function ----
        sampler = TPESampler(seed=random_state, n_startup_trials=n_startup_trials)
        study = optuna.create_study(sampler=sampler, direction="maximize")

        sb = SaveBestOnlyCallback(export_dir)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[sb])


        # ---- save best trial ----
        metadata = {}
        metadata["best_trial"] = f"trial-{study.best_trial.number:0=4d}"
        metadata["score_from_best_trial"] = study.best_trial.value

        best_framework = self.framework_store[study.best_trial.number]
        # print(best_framework)

        # with open(os.path.join(export_dir, "best_framework.yml"), "w") as f:
        #     yaml.dump(best_framework, f)

        # with open(os.path.join(export_dir, "metadata.yml"), "w") as f:
        #     yaml.dump(metadata, f)

        return best_framework, metadata






def test(framework=None,

         df_train=None,
         df_test=None,
         max_length=None,

         sum_of_scores=None,
         misc=None,
         ):


    if framework["name"] == "logit":
        classifier = LogisticRegression(**framework["params"])
    elif framework["name"] == "svm":
        classifier = LinearSVC(**framework["params"])
    elif framework["name"] == "ridge":
        classifier = RidgeClassifier(**framework["params"])



    # ---- setup test dataset ----
    train_index, train_x, train_y = to_embedding(df_train, max_length)
    test_index, test_x, test_y = to_embedding(df_test, max_length)

    # ---- train ----
    clf = classifier.fit(train_x, train_y)

    # ---- test ----
    test_pred_y = clf.predict(test_x)
    if hasattr(clf, "predict_proba"):
        # For models like LogisticRegression
        probability_score = clf.predict_proba(test_x)[:, 1]  # Get probabilities for the positive class
    elif hasattr(clf, "decision_function"):
        # For models like LinearSVC or RidgeClassifier
        probability_score = clf.decision_function(test_x)
    else:
        raise ValueError(
            f"The classifier {type(clf).__name__} does not support probability or decision outputs.")



    # ===================================================
    # =============== result saving =====================
    # ===================================================
    # ---- linear model weights (interpretability) ----
    clf_coef = clf.coef_
    misc["model_weights"].append(clf_coef[0])

    # ---- save indexes ----
    misc["train_index"].append(train_index)
    misc["test_index"].append(test_index)


    # ---- calculate metrics ----
    auroc = roc_auc_score(test_y, probability_score)
    accuracy = accuracy_score(test_y, test_pred_y)
    recall = recall_score(test_y, test_pred_y)
    specificity = specificity_score(test_y, test_pred_y)

    precision = precision_score(test_y, test_pred_y) # positive predictive value
    tn, fp, fn, tp = confusion_matrix(test_y, test_pred_y).ravel()
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0 # negative predictive value

    sum_of_scores["accuracy"].append(accuracy)
    sum_of_scores["auroc"].append(auroc)
    sum_of_scores["recall"].append(recall)
    sum_of_scores["specificity"].append(specificity)
    sum_of_scores["precision"].append(precision)
    sum_of_scores["npv"].append(npv)

    return





class SaveBestOnlyCallback:
    def __init__(self, directory):
        self.directory = directory

    def __call__(self, study: Study, trial: Trial):
        best_trial_id = study.best_trial.number

        for trial in study.trials:
            trial_dir = f"trial-{trial.number:0=4d}"
            directory = os.path.join(self.directory, trial_dir)
            if os.path.exists(directory) and trial.number != best_trial_id:
                shutil.rmtree(directory)


