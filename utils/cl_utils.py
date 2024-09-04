import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from copy import deepcopy

plt.rcParams.update({
    "font.family": "serif",})

class Client:
    def __init__(self, args, loaders, model, optimizer, criterion, memory, client_id, cls_assignment_client):
        super(Client, self).__init__()
        self.args = args
        self.memory = memory
        self.loaders = loaders
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.client_id = client_id

        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]

        skip = args.n_classes_per_task
        task_list = [tuple(cls_assignment_client[i:i+skip]) for i in range(0, args.n_classes, skip)]
        self.cls_assignment = cls_assignment_client
        self.task_list = task_list

        self.seen = 0
        self.seen_per_task = 0
        self.task_id = 0
        self.train_task_loss = 0
        self.train_completed = False
        self.num_batches = 0
        self.train_iterators = [iter(loader) for loader in self.train_loader]
        self.importance_weight = 1.0
        self.global_model = None
        self.last_local_model = deepcopy(model)


        # define the augmentations for multiple training epochs
        flip = transforms.RandomHorizontalFlip()
        rotation = transforms.RandomRotation(degrees=20) 
        augment = torch.nn.Sequential(flip, rotation)
        self.augment = augment


    def get_next_batch(self):
        task_iterator = self.train_iterators[self.task_id]
        try:
            samples, labels = next(task_iterator)
            n_samples = len(labels)
            self.seen_per_task += n_samples
            self.seen += n_samples
            self.num_batches += 1
            self.importance_weight = self.compute_weight()
            return samples, labels
            
        except StopIteration: # task is completed
            # reset counters
            self.seen_per_task = 0
            return None, None


    def compute_weight(self):
        return self.seen_per_task


    def compute_proximal_term(self):
        proximal_term = 0.0
        for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
            proximal_term += (w - w_t).norm(2)
        return proximal_term


    def training_step(self, samples, labels):
        self.optimizer.zero_grad()
        logits = self.model(samples)
        loss = self.criterion(logits, labels)
        if 'fprox' in self.args.fl_update:
            proximal_term = self.compute_proximal_term()
            loss += proximal_term * (self.args.mu / 2)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def compute_loss(self, logger, run):
        loss = self.train_task_loss / self.num_batches
        logger['train']['loss'][self.client_id][self.task_id][run] = loss
        # reset counters
        self.num_batches = 0
        self.train_task_loss = 0
        return logger

    def add_gaussian_noise(self, embedding_matrix, mean=0, std=0.1):
        noise = torch.Tensor(np.random.normal(mean, std, size=embedding_matrix.shape)).to(self.args.device)
        noisy_embedding_matrix = embedding_matrix + noise
        return noisy_embedding_matrix


    def train(self, samples, labels):
        self.model.train()
        samples, labels = samples.to(self.args.device), labels.to(self.args.device)
        batch_loss = self.training_step(samples, labels)
        if self.args.dataset_name in ['newsgroup', 'reuters', 'yahoo', 'dbpedia']:
            # multiple gradient updates for the same mini-batch if local_epochs > 1
            for local_epoch in range(self.args.local_epochs - 1):
                batch_loss = self.training_step(samples , labels)
        else:
            # multiple gradient updates for the same mini-batch if local_epochs > 1
            for local_epoch in range(self.args.local_epochs - 1):
                batch_loss = self.training_step(self.augment(samples) , labels)
        self.train_task_loss += batch_loss


    def train_with_update(self, samples, labels):
        self.model.train()
        samples, labels = samples.to(self.args.device), labels.to(self.args.device)
        current_classes = self.get_current_task()
        batch_loss = self.training_step(samples, labels)
        if self.args.dataset_name in ['newsgroup', 'reuters', 'yahoo', 'dbpedia']:
            # multiple gradient updates for the same mini-batch if local_epochs > 1
            for local_epoch in range(self.args.local_epochs - 1):
                batch_loss = self.training_step(samples, labels)
        else:
            # multiple gradient updates for the same mini-batch if local_epochs > 1
            for local_epoch in range(self.args.local_epochs - 1):
                batch_loss = self.training_step(self.augment(samples), labels)
        self.train_task_loss += batch_loss

        if self.args.update_strategy == 'reservoir':
            self.memory.reservoir_update(samples, labels, self.task_id)
        if self.args.update_strategy == 'balanced':
            self.memory.class_balanced_update(samples, labels, self.task_id, self.model, current_classes)
        if self.args.update_strategy == 'uncertainty':
            self.memory.uncertainty_update(samples, labels, self.task_id, self.model)


    def train_with_memory(self, samples, labels):
        self.model.train()
        samples, labels = samples.to(self.args.device), labels.to(self.args.device)
        current_classes = self.get_current_task()

        if self.args.sampling_strategy == 'uncertainty':
            mem_x, mem_y, _ = self.memory.uncertainty_sampling(self.model, exclude_task=self.task_id,
                                                                subsample_size=self.args.subsample_size)
        if self.args.sampling_strategy == 'random':
            mem_x, mem_y, _ = self.memory.random_sampling(self.args.batch_size, exclude_task=self.task_id)
        if self.args.sampling_strategy == 'balanced_random':
            mem_x, mem_y, _ = self.memory.balanced_random_sampling(self.args.batch_size, exclude_task=self.task_id)

        mem_x, mem_y = mem_x.to(self.args.device), mem_y.to(self.args.device)
        input_x = torch.cat([samples, mem_x]) # .to(self.args.device)
        input_y = torch.cat([labels, mem_y])  # .to(self.args.device)
        batch_loss = self.training_step(input_x, input_y)

        # multiple gradient updates for the same mini-batch if local_epochs > 1
        for local_epoch in range(self.args.local_epochs - 1):
            if self.args.sampling_strategy == 'uncertainty':
                mem_x, mem_y, _ = self.memory.uncertainty_sampling(self.model, exclude_task=self.task_id,
                                                                   subsample_size=self.args.subsample_size)
            if self.args.sampling_strategy == 'random':
                mem_x, mem_y, _ = self.memory.random_sampling(self.args.batch_size, exclude_task=self.task_id)
            if self.args.sampling_strategy == 'balanced_random':
                mem_x, mem_y, _ = self.memory.balanced_random_sampling(self.args.batch_size, exclude_task=self.task_id)

            mem_x, mem_y = mem_x.to(self.args.device), mem_y.to(self.args.device)
            if self.args.dataset_name in ['newsgroup', 'reuters', 'yahoo', 'dbpedia']:
                input_x = torch.cat([samples, mem_x]) # .to(self.args.device)
            else:
                input_x = torch.cat([self.augment(samples), mem_x]) # .to(self.args.device)

            input_y = torch.cat([labels, mem_y])  # .to(self.args.device)
            batch_loss = self.training_step(input_x, input_y)
        
        self.train_task_loss += batch_loss
        if self.args.update_strategy == 'reservoir':
            self.memory.reservoir_update(samples, labels, self.task_id)
        if self.args.update_strategy == 'balanced':
            self.memory.class_balanced_update(samples, labels, self.task_id, self.model, current_classes)


    @torch.no_grad()
    def test(self, logger, run):
        self.model.eval()
        for task_id_eval, eval_loader in enumerate(self.test_loader):
            total_correct, total = 0.0, 0.0
            y_pred = []
            y_true = []
            if task_id_eval > self.task_id:
                break
            for samples, labels in eval_loader:
                samples, labels = samples.to(self.args.device), labels.to(self.args.device)
                logits = self.model(samples)
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum()
                total += len(labels)
                y_true.append(labels)
                y_pred.append(preds)

            y_true = torch.cat(y_true).cpu()
            y_pred = torch.cat(y_pred).cpu()
            accuracy = total_correct/total

            cm = confusion_matrix(y_true, y_pred, labels=self.cls_assignment)
            cm_display = ConfusionMatrixDisplay(cm, display_labels=self.cls_assignment).plot()
            plt.tight_layout()
            plt.title(f'Accuracy: {accuracy:.3f}')
            plt.savefig(f'{self.args.dir_results}client{self.client_id}_run{run}_cm_{self.task_id}_{task_id_eval}.pdf', format='pdf')
            plt.close()
            logger['test']['acc'][self.client_id][run][self.task_id][task_id_eval] = accuracy
        return logger
    

    @torch.no_grad()
    def validation(self, logger, run):
        self.model.eval()
        for task_id_eval, eval_loader in enumerate(self.val_loader):
            total_correct, total = 0.0, 0.0
            if task_id_eval > self.task_id:
                break
            for samples, labels in eval_loader:
                samples, labels = samples.to(self.args.device), labels.to(self.args.device)
                logits = self.model(samples)
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum()
                total += len(labels)
            accuracy = total_correct/total
            logger['val']['acc'][self.client_id][run][self.task_id][task_id_eval] = accuracy
        return logger
    

    @torch.no_grad()
    def balanced_accuracy(self, logger, run):
        self.model.eval()
        y_pred = []
        y_true = []
        for task_id_eval, eval_loader in enumerate(self.test_loader):
            if task_id_eval > self.task_id:
                break
            for samples, labels in eval_loader:
                samples, labels = samples.to(self.args.device), labels.to(self.args.device)
                logits = self.model(samples)
                preds = logits.argmax(dim=1)
                y_true.append(labels)
                y_pred.append(preds)
        y_true = torch.cat(y_true).cpu()
        y_pred = torch.cat(y_pred).cpu()
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)    
        logger['test']['bal_acc'][self.client_id][run] = balanced_accuracy

        cm = confusion_matrix(y_true, y_pred, labels=self.cls_assignment)
        cm_display = ConfusionMatrixDisplay(cm, display_labels=self.cls_assignment).plot()
        plt.tight_layout()
        plt.title(f'Accuracy: {balanced_accuracy:.3f}')
        plt.savefig(f'{self.args.dir_results}client{self.client_id}_run{run}_cm_final.pdf', format='pdf')
        plt.close()

        return logger
    

    def forgetting(self, logger, run):
        '''https://github.com/clovaai/rainbow-memory/blob/master/main.py

        '''
        forget_list_test = []
        forget_list_val = []
        cls_acc_test = logger['test']['acc'][self.client_id][run]
        cls_acc_val = logger['val']['acc'][self.client_id][run]
        for k in range(self.args.n_tasks):
            forget_k_test = []
            forget_k_val = []
            for j in range(self.args.n_tasks):
                if j < k:
                    forget_k_test.append(cls_acc_test[:k, j].max() - cls_acc_test[k, j])
                    forget_k_val.append(cls_acc_val[:k, j].max() - cls_acc_val[k, j])
                else:
                    forget_k_test.append(None)
                    forget_k_val.append(None)
            forget_list_test.append(forget_k_test)
            forget_list_val.append(forget_k_val)
        logger['test']['forget'][self.client_id][run] = np.mean(forget_list_test[-1][:-1])
        logger['val']['forget'][self.client_id][run] = np.mean(forget_list_val[-1][:-1])
        return logger
    
    def get_weight(self):
        return self.importance_weight

    def get_current_task(self):
        return self.task_list[self.task_id]

    def get_parameters(self):
        # return self.model.state_dict()
        return self.model.parameters()

    def save_last_global_model(self, global_model):
        self.global_model = global_model

    def get_last_global_model(self):
        return self.global_model
    
    def save_last_local_model(self):
        self.last_local_model = deepcopy(self.model)

    def get_last_local_model(self):
        return self.last_local_model

    def update_parameters(self, global_parameters):
        # self.model.load_state_dict(global_parameters)
        local_parameters = self.model.state_dict()
        averaged_parameters = {}
        for key in local_parameters.keys():
            averaged_parameters[key] = (local_parameters[key] + global_parameters[key]) / 2.0
        self.model.load_state_dict(averaged_parameters)