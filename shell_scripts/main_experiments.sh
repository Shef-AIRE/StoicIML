export PYTHONWARNINGS="ignore"

# CHARPROTSET, Integer-label, Classifiers
python main_linear-intlabel-protein.py \
    --config configs/main_exp/VLP_200.yaml \
    --model "logit"

python main_linear-intlabel-protein.py \
    --config configs/main_exp/VLP_200.yaml \
    --model "ridge"

python main_linear-intlabel-protein.py \
    --config configs/main_exp/VLP_200.yaml \
    --model "svm"



# CHARPROTSET, One-hot, Classifiers
python main_linear-onehot-protein.py \
    --config configs/main_exp/VLP_200.yaml \
    --model "logit"

python main_linear-onehot-protein.py \
    --config configs/main_exp/VLP_200.yaml \
    --model "ridge"

python main_linear-onehot-protein.py \
    --config configs/main_exp/VLP_200.yaml \
    --model "svm"


# Clusters, Integer-label, Classifiers
python main_linear-intlabel-cluster.py \
    --config configs/main_exp/VLP_200.yaml \
    --model "logit"

python main_linear-intlabel-cluster.py \
    --config configs/main_exp/VLP_200.yaml \
    --model "ridge"

python main_linear-intlabel-cluster.py \
    --config configs/main_exp/VLP_200.yaml \
    --model "svm"


# Clusters, One-hot, Classifiers
python main_linear-onehot-cluster.py \
    --config configs/main_exp/VLP_200.yaml \
    --model "logit"

python main_linear-onehot-cluster.py \
    --config configs/main_exp/VLP_200.yaml \
    --model "ridge"

python main_linear-onehot-cluster.py \
    --config configs/main_exp/VLP_200.yaml \
    --model "svm"

