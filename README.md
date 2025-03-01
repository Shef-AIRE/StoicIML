# Classifying the Stoichiometry of Virus-like Particles with Interpretable Machine Learning

## Introduction 
This repository contains the scikit-learn implementation of the **StoicIML** pipeline, as shown below. **StoicIML** is an an interpretable, data-driven pipeline that leverages linear machine learning models to classify protein stoichiometry in virus-like particle assembly. 
## Pipeline
![VLP-workflow](image/VLP-workflow-7.png)

## Datasets
The ``datasets/curated``folder contains 200 protein sequences that assemble into either 60-mer or 180-mer VLPs, sourced from the [RCSB PDB](https://www.rcsb.org/) [1]. 

## Requirements
Please install (a) the packages listed in ``requirements.txt`` and (b) the [feature selection repository](https://github.com/jundongl/scikit-feature?tab=readme-ov-file) [2].
```
pip install -r requirements.txt
```

## Reproduce Results
* The basic configurations are in``configs/congigs.py``, main experiment configs in ``configs/main_exp/VLP_200.yaml``, and ablation study configs in ``configs/study1_truncate/*.yaml`` and ``configs/study2_position_selection/*.yaml``.

For the main experiments, you can directly run the following command.
```
chmod +x ./shell_scripts/main_experiments.sh
./shell_scripts/main_experiments.sh
```
For the ablation study, you can run
```
chmod +x ./shell_scripts/ablation_study.sh
./shell_scripts/ablation_study.sh
```

## Citation
Please cite our paper if you find it useful.
```
@misc{zhang2025classifyingstoichiometryviruslikeparticles,
      title={Classifying the Stoichiometry of Virus-like Particles with Interpretable Machine Learning}, 
      author={Jiayang Zhang and Xianyuan Liu and Wei Wu and Sina Tabakhi and Wenrui Fan and Shuo Zhou and Kang Lan Tee and Tuck Seng Wong and Haiping Lu},
      year={2025},
      eprint={2502.12049},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.12049}, 
}
```

## References
```
[1] Berman HM, Westbrook J, Feng Z, Gilliland G, Bhat TN, Weissig H, Shindyalov IN, Bourne PE. The protein data bank. Nucleic acids research. 2000 Jan 1;28(1):235-42.
[2] Li J, Cheng K, Wang S, Morstatter F, Trevino RP, Tang J, Liu H. Feature selection: A data perspective. ACM computing surveys (CSUR). 2017 Dec 6;50(6):1-45.
```
