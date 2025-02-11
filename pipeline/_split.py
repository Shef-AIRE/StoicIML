
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import StratifiedShuffleSplit
from typing import Union
class RandomSplit:
    def __init__(self, df:pd.DataFrame, ratio:tuple, seed=None, stratify_column_name=None):
        self.df = df
        self.ratio = ratio
        self.seed = seed
        self.stratify_column_name = stratify_column_name

    def split(self):
        """
        Return:
        -------
        df_train: pd.DataFrame
            The training set.
        df_val: pd.DataFrame
            The validation set.
        df_test: pd.DataFrame
            The test set.
        Only return the above dataframes once.

        Notes:
        ------
        - The sum of the ratios must be 10. For example, [7, 1.5, 1.5] means 70% training, 15% validation, and 15% test.
        - If `test` is set to 0, the validation set will be used as the test set.
        """
        train, val, test = self.ratio
        total = train + val + test

        if total != 10:
            raise ValueError("The sum of ratio must be 10.")

        df_train, df_val = train_test_split(self.df, test_size=((val + test) / total),
                                            random_state=self.seed, shuffle=True, stratify=self.df[self.stratify_column_name])
        if test == 0:
            df_test = df_val

        else:
            df_val, df_test = train_test_split(df_val, test_size=(test / (val + test)),
                                               random_state=self.seed, shuffle=True, stratify=df_val[self.stratify_column_name])

        yield df_train, df_val, df_test




class KFoldSplit:
    def __init__(self, df:pd.DataFrame, n_folds:int, seed=None, stratify_column_name=None):
        self.df = df
        self.n_folds = n_folds
        self.seed = seed
        self.stratify_column_name = stratify_column_name

    def split(self):
        skf = StratifiedKFold(n_splits=self.n_folds, random_state=self.seed, shuffle=True)
        for idx_train, idx_test in skf.split(X=self.df, y=self.df[self.stratify_column_name]):
            # df_train, df_test = self.df.iloc[idx_train], self.df.iloc[idx_test]
            yield idx_train, idx_test   # return the indices of the train and test set




