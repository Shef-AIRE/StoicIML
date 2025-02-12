export PYTHONWARNINGS="ignore"

## ----- truncation by length: Ridge # positions-----
python main_linear-onehot-protein.py \
  --config configs/study1_truncate/Truncate_171.yaml \
  --model "ridge"


## ----- selection by weights: Ridge a% positions-----
python main_linear-study2-onehot-protein.py \
  --config configs/study2_position_selection/study2_weights.yaml \
  --model "ridge" \
  --positional_index "main-exp-result/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9/capped_indices_perc6.npy"


## ----- selection by laplacian score: Ridge a% positions-----
python main_linear-study2-onehot-protein.py \
  --config configs/study2_position_selection/study2_laplacian.yaml \
  --model "ridge" \
  --positional_index "main-exp-result/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9/laplacian_indices_perc27.npy"


## ----- selection by variance: Ridge a% positions-----
python main_linear-study2-onehot-protein.py \
  --config configs/study2_position_selection/study2_variance.yaml \
  --model "ridge" \
  --positional_index "main-exp-result/trial-result-VLP200-ridge-onehotprotein-seqlen1426-optimfold9/variance_indices_perc24.npy"

