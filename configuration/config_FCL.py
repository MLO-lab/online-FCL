import argparse

def base_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--framework",
        type=str,
        default='FCL',
        help="Framework to evaluate [CL]",
    )

    parser.add_argument(
        "--dir_data",
        type=str,
        default='./data/',
        help="Directory to save the datasets",
    )

    parser.add_argument(
        "--dir_output",
        type=str,
        default='./rebuttal/',
        help="Directory to save the output",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default='cifar10',
        help="Name of the dataset",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default='default',
        help="Model architecture [mlp, resnet, default]",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default='sgd',
        help="Optimizer [sgd, adam]",
    )

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Number of epochs at batch-level (multiple gradient updates per batch)",
    )

    parser.add_argument(
        "--n_runs",
        type=int,
        default=3,
        help="Number of runs for each experiment",
    )

    parser.add_argument(
        "--n_tasks", 
        type=int, 
        default=-1, 
        help="Number of tasks (-1 default number of tasks for each dataset)"
    )

    parser.add_argument(
        "--with_memory",
        type=int,
        default=1,
        help="Use of episodic memory [0 (no), 1 (yes)]",
    )

    parser.add_argument(
        "--memory_size",
        type=int,
        default=500,
        help="Size of episodic memory [200, 500, 1000]",
    )

    parser.add_argument(
        "--update_strategy",
        type=str,
        default='balanced',
        help="Memory update strategy [reservoir, balanced]",
    )

    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default='random',
        help="Memory sampling strategy [random, uncertainty]",
    )


    parser.add_argument(
        "--balanced_update",
        type=str,
        default='random',
        help="Update strategy for class-balanced memory management [random, uncertainty]",
    )

    parser.add_argument(
        "--uncertainty_score",
        type=str,
        default='bregman',
        help="Uncertainty metric for uncertainty management (bregman, confidence, margin, entropy, rainbow)",
    )

    parser.add_argument(
        "--subsample_size",
        type=int,
        default=50,
        help="Size of the subsample (taken from the memory for replay) for computing the uncertainty scores when sampling from the memory",
    )    


    parser.add_argument(
        "--balanced_step",
        type=str,
        default='bottomk',
        help="Sampling strategy for uncertainty-based class-balanced memory management (step (step-sized), topk (top-k), bottomk (bottom-k))",
    )

    parser.add_argument(
        "--n_clients",
        type=int,
        default=5,
        help="Number of clients",
    )

    ####################### FEDERATED PARAMETERS ###########################
    parser.add_argument(
        "--overlap",
        type=str,
        default='non-overlap',
        help="Overlapping tasks across clients (overlap, non-overlap)",
    )

    parser.add_argument(
        "--burnin",
        type=int,
        default=30,
        help="Burnin time (number of epochs) before a client contributes to the communication rounds",
    )

    parser.add_argument(
        "--jump",
        type=int,
        default=5,
        help="Number of epochs to jump before a client contributes to the communication rounds",
    )

    parser.add_argument(
        "--fl_update",
        type=str,
        default='w_favg',
        help="Memory sampling strategy [favg (FedAvg), w_favg (weighted FedAvg), fprox (FedProx), w_fprox (weighted fprox)]",
    )

    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
        help="Parameter for proximal term in FedProx",
    )    

    args = parser.parse_args()
    return args