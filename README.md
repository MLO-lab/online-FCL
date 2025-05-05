# Federated Continual Learning Goes Online (Online-FCL)

The repository contains the code to reproduce the results of the paper ["Federated Continual Learning Goes Online: Uncertainty-Aware Memory Management for Vision Tasks and Beyond"](https://openreview.net/forum?id=f65RuQgVlp). 

### Requirements
The codebase is implemented in Python 3.12.2 and tested on Linux OS.
 - `pytorch==2.2.2`
 - `torchvision==0.17.2`
 - `transformers==4.40.2`
 - `numpy==1.26.4`
 - `scikit-learn==1.4.1`
 - `medmnist==3.0.1`
 - `matplotlib==3.8.0` 

### Structure of the repo
- `utils` contains all the utilities to preprocess and save the data, to run the experiments, and to save the results.
- `models` contains the model architectures of our evaluation framework.
- `configuration` contains the hyperparameters (dataset, memory population strategy, memory sampling strategy, memory size, etc.) to run our experiments.

### Run the experiments
The current version of the configuration file (`configuration/config_jup.py`) enables running an experiment for five different clients on CIFAR10 over three runs. All the parameters can be changed as desired in the `main_OFCL.ipynb` file.
The most important hyperparameters are the followings.

For the continual learning and memory management part:
- `--local_epochs`: to perform multiple gradient updates per mini-batch (number of local epochs).
- `--memory_size`: to set the size of the memory buffer.
- `--dataset_name`: name of the dataset to evaluate.
- `--update_strategy`: to select whether we want to use reservoir (`reservoir`) or class-balanced update (`balanced`).
- `--balanced_update`: if `balanced` is used, we can select whether we want to populate the memory randomly (`random`) or use the uncertainty (`uncertainty`).
- `--uncertainty_score`: if `uncertainty` is used, we can decide the uncertainty score we want to use among the available ones (`bregman`, `confidence`, `margin`, `entropy`, `rainbow`, `ratio`).
- `--balanced_step`: if `uncertainty` is used, we can decide the sorting strategy, i.e., `bottomk` (bottom-k), `step` (step-sized), and `topk` (top-k).

For the federated learning part:
- `--overlap`: to set if the clients works on overlapping tasks (`overlap` (default)) or not (`non-overlap`).
- `--burnin`: number of epochs before a client contributes to the communication rounds.
- `--jump`: number of epochs to skip before a client contributes to the communication rounds.
- `--fl_update`: FedAvg (`favg`), weighted FedAvg (`w_favg`).

### Citation
```
@inproceedings{
serra2025federated,
title={Federated Continual Learning Goes Online: Uncertainty-Aware Memory Management for Vision Tasks and Beyond},
author={Giuseppe Serra and Florian Buettner},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=f65RuQgVlp}
}
```