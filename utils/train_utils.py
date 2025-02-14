import os, pickle
import torch
import numpy as np
from collections import defaultdict
from models.resnet import SlimResNet18
from models.mlp import MLP
from sklearn.metrics import balanced_accuracy_score
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from utils.data_loader import get_statistics
from utils.cl_utils import Client
from utils.utils_memory import Memory

def get_free_gpu_idx():
    """Get the index of the GPU with current lowest memory usage."""
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used  > ./output/tmp")
    memory_available = [int(x.split()[2]) for x in open("./output/tmp", "r").readlines()]
    return np.argmin(memory_available)


def get_logger(args):
    _, _, _, _, _ = get_statistics(args)

    log = {}
    log['train'] = defaultdict(dict)
    for client_id in range(args.n_clients):
        log['train']['loss'][client_id] = np.zeros([args.n_tasks, args.n_runs])

    for mode in ['test', 'val']:
        log[mode] = defaultdict(dict)
        for client_id in range(args.n_clients):
            log[mode]['acc'][client_id] = np.zeros([args.n_runs, args.n_tasks, args.n_tasks])
            log[mode]['forget'][client_id] = np.zeros([args.n_runs])
            log[mode]['bal_acc'][client_id] = np.zeros([args.n_runs])

    log['global_test'] = defaultdict(dict)
    log['global_test']['bal_acc'] = np.zeros([args.n_runs])
    for task_id in range(args.n_tasks):
        log['global_test']['acc'] = np.zeros([args.n_runs, args.n_tasks, args.n_tasks])

    return log


def custom_resnet(args, model):
    model.conv1 = torch.nn.Conv2d(args.input_size[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, args.n_classes)
    return model.to(args.device)


def initialize_model(args):
    if args.model_name == 'resnet18':
        model = resnet18().to(args.device)
    if args.model_name == 'resnet18_pre':
        resnet = resnet18(weights='DEFAULT')
        model = custom_resnet(args, resnet)
    if args.model_name == 'resnet50':
        model = resnet50().to(args.device)
    if args.model_name == 'resnet50_pre':
        resnet = resnet50(weights='DEFAULT')
        model = custom_resnet(args, resnet)
    if args.model_name == 'resnet':
        model = SlimResNet18(nclasses=args.n_classes, input_size=args.input_size).to(args.device)
    if args.model_name == 'mlp':
        # model = MLP(in_channels=args.input_size, hidden_channels=[256,128,64,32,args.n_classes], dropout=0.4).to(args.device)
        model = MLP(args.input_size, [512], args.n_classes).to(args.device)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.004) # 3e-4 (yahoo 1e-2, newsgroup 0.004, dbpedia)

    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, criterion


def initialize_clients(args, loader_clients, cls_assignment_list, run):
    clients = []
    for client_id in range(args.n_clients):
        loader_client = loader_clients[client_id]
        cls_assignment_client = cls_assignment_list[client_id]
        # for reproducibility purposes
        np.random.seed(run)
        torch.manual_seed(run)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
            
        model, optimizer, criterion = initialize_model(args)
        memory_client = Memory(args)
        client = Client(args, loader_client, model, optimizer, criterion, memory_client, client_id, cls_assignment_client)
        clients.append(client)

    # initialize global model
    global_model = modelAvg(args, list_models=[client.model for client in clients])
    for client in clients:
        client.global_model = global_model

    return clients


def FedAvg(args, selected_clients, clients):
    global_model, _, _ = initialize_model(args)

    global_params = list(global_model.parameters())
    client_weights = [clients[client_id].get_weight() for client_id in selected_clients]
    total_weights = sum(client_weights)
    client_weights = [weight / total_weights for weight in client_weights]
    client_params = [list(clients[client_id].get_parameters()) for client_id in selected_clients]

    weighted_params = []
    for i in range(len(global_params)):
        weighted_param = torch.zeros_like(global_params[i])
        for j in range(len(client_params)):
            weighted_param.data += client_weights[j] * client_params[j][i].data
        weighted_params.append(weighted_param)

    for global_param, weighted_param in zip(global_params, weighted_params):
        global_param.data = weighted_param

    last_global_model = None
    for client_id in selected_clients:
        last_global_model_tmp = clients[client_id].get_last_global_model()
        if last_global_model_tmp is not None:
            last_global_model = last_global_model_tmp
            break

    if last_global_model == None:
        return global_model
    else:
        return modelAvg(args, [last_global_model, global_model])


def weightedFedAvg(args, selected_clients, clients):
    count_clients_per_class = defaultdict(list)
    for client_id in selected_clients:
        current_classes = clients[client_id].get_current_task()
        for class_id in current_classes:
            count_clients_per_class[class_id].append(client_id)

    class_models = []
    for class_id in count_clients_per_class.keys():
        if len(count_clients_per_class[class_id]) > 1:
            avg_class_model = FedAvg(args, count_clients_per_class[class_id], clients)
            class_models.append(avg_class_model)
        else:
            client_id = count_clients_per_class[class_id][0]
            class_models.append(clients[client_id].model)

    global_model = modelAvg(args, class_models)
    last_global_model = None
    for client_id in selected_clients:
        last_global_model_tmp = clients[client_id].get_last_global_model()
        if last_global_model_tmp is not None:
            last_global_model = last_global_model_tmp
            break

    if last_global_model == None:
        return global_model
    else:
        return modelAvg(args, [last_global_model, global_model])


def modelAvg(args, list_models):
    global_model, _, _ = initialize_model(args)

    global_params = list(global_model.parameters())
    single_params = [list(single_model.parameters()) for single_model in list_models]
    stacked_params = [torch.stack(param_list) for param_list in zip(*single_params)]
    averaged_params = [param.mean(dim=0) for param in stacked_params]
    for global_param, averaged_param in zip(global_params, averaged_params):
        global_param.data = averaged_param

    return global_model


@torch.no_grad()
def test_global_model(args, test_loader, model, logger, run, mode='global_test'):
    task_id = args.n_tasks - 1
    model.eval()
    y_true = []
    y_pred = []
    for task_id_eval, eval_loader in enumerate(test_loader):
        total_correct, total = 0.0, 0.0
        if task_id_eval > task_id:
            break

        for samples, labels in eval_loader:
            samples, labels = samples.to(args.device), labels.to(args.device)
            logits = model(samples)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum()
            total += len(labels)
            y_true.append(labels)
            y_pred.append(preds)

        accuracy = total_correct/total
        logger[mode]['acc'][run][task_id][task_id_eval] = accuracy

    y_true = torch.cat(y_true).cpu()
    y_pred = torch.cat(y_pred).cpu()
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)    
    logger[mode]['bal_acc'][run] = balanced_accuracy

    return logger


def save_results(args, logger):
    logger_fn = f'{args.dir_results}/logger.pkl'
    with open(logger_fn, 'wb') as outfile:
        pickle.dump(logger, outfile)
        outfile.close()

def compute_avg_acc_for(args, logger):
    final_accs = []
    final_fors = []
    for client_id in range(args.n_clients):
        final_acc = np.mean(np.mean(logger["test"]["acc"][client_id], 0)[args.n_tasks-1,:], 0)
        final_for = np.mean(logger["test"]["forget"][client_id])
        final_accs.append(final_acc)
        final_fors.append(final_for)
    return np.mean(final_accs), np.std(final_accs), np.mean(final_fors), np.std(final_fors)