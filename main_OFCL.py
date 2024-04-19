import torch
import numpy as np

from configuration import config_FCL
from utils.data_loader import get_loader_all_clients
from utils.train_utils import get_free_gpu_idx, get_logger, initialize_clients, FedAvg, weightedFedAvg, test_global_model, save_results


args = config_FCL.base_parser()
logger = get_logger(args)

if torch.cuda.is_available():
    gpu_idx = get_free_gpu_idx()
    args.cuda = True
    args.device = f'cuda:{gpu_idx}'
else:
    args.device = 'cpu' 
print(args)

for run in range(args.n_runs):
    loader_clients, cls_assignment_list, global_test_loader = get_loader_all_clients(args, run)
    clients = initialize_clients(args, loader_clients, cls_assignment_list, run)

    while not all([client.train_completed for client in clients]):
        for client in clients:
            if not client.train_completed:
                samples, labels = client.get_next_batch()
                if samples is not None:
                    if args.with_memory:
                        if client.task_id == 0:
                            client.train_with_update(samples, labels)
                        else:
                            client.train_with_memory(samples, labels)
                    else:
                        client.train(samples, labels)
                else:
                    print(f'Run {run} - Client {client.client_id} - Task {client.task_id} completed - {client.get_current_task()}')
                    # compute loss train
                    logger = client.compute_loss(logger, run)
                    print(f'Run {run} - Client {client.client_id} - Test time - Task {client.task_id}')
                    logger = client.test(logger, run)
                    logger = client.validation(logger, run)
                    logger = client.forgetting(logger, run)
                    if client.task_id + 1 >= args.n_tasks:
                        client.train_completed = True
                        print(f'Run {run} - Client {client.client_id} - Train completed')
                        logger = client.balanced_accuracy(logger, run)
                    else:
                        client.task_id += 1

        # COMMUNICATION ROUND PART
        selected_clients = [client.client_id for client in clients if (client.num_batches >= args.burnin and client.num_batches % args.jump == 0 and client.train_completed == False)]
        if len(selected_clients) > 1:
            # communication round when all clients process a mini-batch
            if args.fl_update == 'favg':
                global_model = FedAvg(args, selected_clients, clients)
            if args.fl_update == 'w_favg':
                global_model = weightedFedAvg(args, selected_clients, clients)

            global_parameters = global_model.state_dict()
            # local models update with averaged global parameters
            for client_id in selected_clients:
                clients[client_id].update_parameters(global_parameters)
                clients[client_id].save_last_global_model(global_model)

    # global model accuracy when all clients finish their training on all tasks (FedCIL ICLR2023)
    logger = test_global_model(args, global_test_loader, global_model, logger, run)

final_accs = []
final_fors = []
for client_id in range(args.n_clients):
    print(f'Client {client_id}: {clients[client_id].task_list}')
    print(np.mean(logger['test']['acc'][client_id], 0))
    final_acc = np.mean(np.mean(logger["test"]["acc"][client_id], 0)[args.n_tasks-1,:], 0)
    final_for = np.mean(logger["test"]["forget"][client_id])
    final_accs.append(final_acc)
    final_fors.append(final_for)
    print(f'Final client accuracy: {final_acc}')
    print(f'Final client forgetting: {final_for}')
    print(f'Final client balanced accuracy: {np.mean(logger["test"]["bal_acc"][client_id])}')
    print()

print(f'Final average accuracy: {np.mean(final_accs):0.4f} (+-) {np.std(final_accs):0.4f}')
print(f'Final average forgetting: {np.mean(final_fors):0.4f} (+-) {np.std(final_fors):0.4f}')

# save training results
save_results(args, logger)