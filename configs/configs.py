from yacs.config import CfgNode as CN

_C = CN()

# Cross-validation
_C.CV = CN()
_C.CV.REPEATS = 10 # random seed & number of repeats
_C.CV.N_FOLD = 10 # number of folds for (train-val : test)
_C.CV.OPTIM_FOLD = 9 # number of folds for hyperparameter optimisation, i.e. number of folds for (train : val)


# Model
_C.MODEL = CN()
# _C.MODEL.TYPE = None    # can be: "svm", "logit", "ridge"
_C.MODEL.SEED = 5


# Sequence
_C.SEQ = CN()
_C.SEQ.INPUT_PATH = "./datasets/curated/VLP_200.csv"
_C.SEQ.MAX_LEN = None # 1426

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result"


def get_cfg_defaults():
    return _C.clone()
