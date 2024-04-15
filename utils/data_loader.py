import os, pickle
import torch
import medmnist
import numpy as np

from torchvision import datasets, transforms

# taken from https://github.com/clovaai/rainbow-memory/blob/master/utils/data_loader.py
def get_statistics(args):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    dataset = args.dataset_name
    if args.dataset_name == 'cifar10LT':
        dataset = 'cifar10'
    if args.dataset_name == 'cifar100LT':
        dataset = 'cifar100'
    if args.dataset_name == 'tinyimagenetLT':
        dataset = 'tinyimagenet'
        
    assert dataset in [
        "mnist",
        "KMNIST",
        "EMNIST",
        "FashionMNIST",
        "SVHN",
        "cifar10",
        "cifar100",
        "CINIC10",
        "imagenet100",
        "imagenet1000",
        "tinyimagenet",
        "bloodmnist",
        "tissuemnist",
        "pathmnist",
        "organamnist",
        "organcmnist",
        "organsmnist",

    ]
    mean = {
        "mnist": (0.1307,),
        "KMNIST": (0.1307,),
        "EMNIST": (0.1307,),
        "FashionMNIST": (0.1307,),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "CINIC10": (0.47889522, 0.47227842, 0.43047404),
        "tinyimagenet": (0.4802, 0.4481, 0.3975), # (0.4914, 0.4822, 0.4465) https://github.com/AlbinSou/ocl_survey/blob/main/src/factories/benchmark_factory.py
        "imagenet100": (0.485, 0.456, 0.406),
        "imagenet1000": (0.485, 0.456, 0.406),
        "bloodmnist": (0.7943, 0.6597, 0.6962),
        "tissuemnist": (0.1020,),
        "pathmnist": (0.7405, 0.5330, 0.7058),
        "organamnist": (0.4678,),
        "organcmnist": (0.4942,),
        "organsmnist": (0.4953,),
    }

    std = {
        "mnist": (0.3081,),
        "KMNIST": (0.3081,),
        "EMNIST": (0.3081,),
        "FashionMNIST": (0.3081,),
        "SVHN": (0.1969, 0.1999, 0.1958),
        # "cifar10": (0.2023, 0.1994, 0.2010), (values taken from the rainbow repo, but they are wrong)
        "cifar10": (0.2470, 0.2435, 0.2616),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "CINIC10": (0.24205776, 0.23828046, 0.25874835),
        "tinyimagenet": (0.2302, 0.2265, 0.2262), #  (0.2023, 0.1994, 0.2010) https://github.com/AlbinSou/ocl_survey/blob/main/src/factories/benchmark_factory.py
        "imagenet100": (0.229, 0.224, 0.225),
        "imagenet1000": (0.229, 0.224, 0.225),
        "bloodmnist": (0.2156, 0.2416, 0.1179),
        "tissuemnist": (0.1000,),
        "pathmnist": (0.1237, 0.1768, 0.1244),
        "organamnist": (0.2975,),
        "organcmnist": (0.2834,),
        "organsmnist": (0.2826,),
    }

    classes = {
        "mnist": 10,
        "KMNIST": 10,
        "EMNIST": 49,
        "FashionMNIST": 10,
        "SVHN": 10,
        "cifar10": 10,
        "cifar100": 100,
        "CINIC10": 10,
        "tinyimagenet": 200,
        "imagenet100": 100,
        "imagenet1000": 1000,
        "bloodmnist": 8,
        "tissuemnist": 8,
        "pathmnist": 8,
        "organamnist": 10,
        "organcmnist": 10,
        "organsmnist": 10,
    }

    in_channels = {
        "mnist": 1,
        "KMNIST": 1,
        "EMNIST": 1,
        "FashionMNIST": 1,
        "SVHN": 3,
        "cifar10": 3,
        "cifar100": 3,
        "CINIC10": 3,
        "tinyimagenet": 3,
        "imagenet100": 3,
        "imagenet1000": 3,
        "bloodmnist": 3,
        "tissuemnist": 1,
        "pathmnist": 3,
        "organamnist": 1,
        "organcmnist": 1,
        "organsmnist": 1,
    }

    inp_size = {
        "mnist": 28,
        "KMNIST": 28,
        "EMNIST": 28,
        "FashionMNIST": 28,
        "SVHN": 32,
        "cifar10": 32,
        "cifar100": 32,
        "CINIC10": 32,
        "tinyimagenet": 64,
        "imagenet100": 224,
        "imagenet1000": 224,
        "bloodmnist": 28,
        "tissuemnist": 28,
        "pathmnist": 28,
        "organamnist": 28,
        "organcmnist": 28,
        "organsmnist": 28,
    }

    if dataset in ['bloodmnist', 'pathmnist', 'tissuemnist']:
        args.n_tasks = 4 if args.n_tasks == -1 else args.n_tasks
    if dataset == 'tinyimagenet':
        args.n_tasks = 20 if args.n_tasks == -1 else args.n_tasks
    else:
        args.n_tasks = 5 if args.n_tasks == -1 else args.n_tasks

    args.input_size = (in_channels[dataset], inp_size[dataset], inp_size[dataset])
    args.n_classes = classes[dataset]
    args.n_classes_per_task = args.n_classes // args.n_tasks

    if args.model_name == 'default':
        if args.dataset_name == 'mnist':
            args.model_name = 'mlp'
        else:
            args.model_name = 'resnet'

    dir_framework = f'{args.dir_output}/{args.framework}/{args.dataset_name}/{args.fl_update}/{args.overlap}/{args.burnin}/{args.jump}'
    if args.optimizer == 'sgd':
        dir_results = f'{dir_framework}/{args.model_name}/{args.optimizer}/{str(args.lr).replace(".","")}'
    else:
        dir_results = f'{dir_framework}/{args.model_name}/{args.optimizer}/'


    if args.update_strategy == 'balanced':
        if args.balanced_update == 'random':
            args.dir_results = f'{dir_results}/{args.memory_size}/{args.batch_size}/{args.local_epochs}/{args.sampling_strategy}/{args.update_strategy}_{args.balanced_update}/'
        else:
            args.dir_results = f'{dir_results}/{args.memory_size}/{args.batch_size}/{args.local_epochs}/{args.sampling_strategy}/{args.update_strategy}_{args.balanced_update}/{args.uncertainty_score}/{args.balanced_step}/'
    if args.update_strategy == 'reservoir':
        args.dir_results = f'{dir_results}/{args.memory_size}/{args.batch_size}/{args.local_epochs}/{args.sampling_strategy}/{args.update_strategy}/'

    if not os.path.exists(args.dir_results):
        os.makedirs(args.dir_results)
        
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        inp_size[dataset],
        in_channels[dataset],        
    )


def get_data(args):
    mean, std, n_classes, inp_size, in_channels = get_statistics(args)

    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    dir_data = f'{args.dir_data}/raw/'
    if args.dataset_name == 'mnist':
        train = datasets.MNIST(dir_data, train=True,  download=True, transform=data_transforms)
        test = datasets.MNIST(dir_data, train=False,  download=True, transform=data_transforms)
        val = None

    if args.dataset_name == 'cifar10':
        train = datasets.CIFAR10(dir_data, train=True,  download=True, transform=data_transforms)
        test = datasets.CIFAR10(dir_data, train=False,  download=True, transform=data_transforms)
        val = None

    if args.dataset_name == 'cifar100':
        train = datasets.CIFAR100(dir_data, train=True,  download=True, transform=data_transforms)
        test = datasets.CIFAR100(dir_data, train=False,  download=True, transform=data_transforms)
        val = None

    if args.dataset_name == 'bloodmnist':
        train = medmnist.BloodMNIST(split='train', download=True, root=dir_data, transform=data_transforms)
        test = medmnist.BloodMNIST(split='test', download=True, root=dir_data, transform=data_transforms)
        val = medmnist.BloodMNIST(split='val', download=True, root=dir_data, transform=data_transforms)

    if args.dataset_name == 'tissuemnist':
        train = medmnist.TissueMNIST(split='train', download=True, transform=data_transforms, root=dir_data)
        test = medmnist.TissueMNIST(split='test', download=True, transform=data_transforms, root=dir_data)
        val = medmnist.TissueMNIST(split='val', download=True, transform=data_transforms, root=dir_data)

    if args.dataset_name == 'pathmnist':
        train = medmnist.PathMNIST(split='train', download=True, transform=data_transforms, root=dir_data)
        test = medmnist.PathMNIST(split='test', download=True, transform=data_transforms, root=dir_data)
        val = medmnist.PathMNIST(split='val', download=True, transform=data_transforms, root=dir_data)
    
    if args.dataset_name == 'organamnist':
        train = medmnist.OrganAMNIST(split='train', download=True, transform=data_transforms, root=dir_data)
        test = medmnist.OrganAMNIST(split='test', download=True, transform=data_transforms, root=dir_data)
        val = medmnist.OrganAMNIST(split='val', download=True, transform=data_transforms, root=dir_data)

    if args.dataset_name == 'organcmnist':
        train = medmnist.OrganCMNIST(split='train', download=True, transform=data_transforms, root=dir_data)
        test = medmnist.OrganCMNIST(split='test', download=True, transform=data_transforms, root=dir_data)
        val = medmnist.OrganCMNIST(split='val', download=True, transform=data_transforms, root=dir_data)

    if args.dataset_name == 'organsmnist':
        train = medmnist.OrganSMNIST(split='train', download=True, transform=data_transforms, root=dir_data)
        test = medmnist.OrganSMNIST(split='test', download=True, transform=data_transforms, root=dir_data)
        val = medmnist.OrganSMNIST(split='val', download=True, transform=data_transforms, root=dir_data)

    if args.dataset_name == 'tinyimagenet':
        dataset = datasets.ImageFolder(f'{args.dir_data}raw/tiny-imagenet-200/train', transform=data_transforms)
        test = datasets.ImageFolder(f'{args.dir_data}raw/tiny-imagenet-200/val', transform=data_transforms)
        train, val = torch.utils.data.random_split(dataset, [80000, 20000])
        test, val = torch.utils.data.random_split(val, [10000, 10000])
        
    return train, test, val


# taken from https://github.com/optimass/Maximally_Interfered_Retrieval/blob/master/data.py
def make_valid_from_train(dataset, cut=0.95):
    tr_ds, val_ds = [], []
    for task_ds in dataset:
        x_t, y_t = task_ds

        # shuffle before splitting
        perm = torch.randperm(len(x_t))
        x_t, y_t = x_t[perm], y_t[perm]

        split = int(len(x_t) * cut)
        x_tr, y_tr   = x_t[:split], y_t[:split]
        x_val, y_val = x_t[split:], y_t[split:]

        tr_ds  += [(x_tr, y_tr)]
        val_ds += [(x_val, y_val)]

    return tr_ds, val_ds


def get_data_per_class(args):
    train, test, val = get_data(args)
    # iterate over train/test to apply the transformations and get images and labels
    train_x = []
    train_y = []
    for img, label in train:
        train_x.append(img)
        if args.dataset_name in medmnist.INFO.keys():
            train_y.append(torch.tensor(label[0], dtype=torch.int64))
        else:
            train_y.append(torch.tensor(label))

    test_x = []
    test_y = []
    for img, label in test:
        test_x.append(img)
        if args.dataset_name in medmnist.INFO.keys():
            test_y.append(torch.tensor(label[0], dtype=torch.int64))
        else:
            test_y.append(torch.tensor(label))
    
    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)
    test_x = torch.stack(test_x)
    test_y = torch.stack(test_y)

    # sort according to the label
    out_train = [(x,y) for (x,y) in sorted(zip(train_x, train_y), key=lambda v : v[1])]
    out_test = [(x,y) for (x,y) in sorted(zip(test_x, test_y), key=lambda v : v[1])]
    train_x, train_y = [torch.stack([elem[i] for elem in out_train]) for i in [0,1]]
    test_x,  test_y  = [torch.stack([elem[i] for elem in out_test]) for i in [0,1]]
    train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)
    test_x, test_y = torch.Tensor(test_x), torch.Tensor(test_y)

    # get indices of class split
    train_idx = (torch.nonzero(train_y[1:] != train_y[:-1], as_tuple=False)[:,0] + 1).tolist()
    test_idx = (torch.nonzero(test_y[1:] != test_y[:-1], as_tuple=False)[:,0] + 1).tolist()
    train_idx = list(zip([0] + train_idx, train_idx + [None]))
    test_idx = list(zip([0] + test_idx, test_idx + [None]))

    train_ds, test_ds = [], []
    for class_id in range(0, args.n_classes):
        tr_s, tr_e = train_idx[class_id]
        te_s, te_e = test_idx[class_id]
        train_ds += [(train_x[tr_s:tr_e], train_y[tr_s:tr_e])]
        test_ds  += [(test_x[te_s:te_e],  test_y[te_s:te_e])]

    if val == None:
        train_ds, val_ds = make_valid_from_train(train_ds)
    else:
        val_x = []
        val_y = []
        for img, label in val:
            val_x.append(img)
            if args.dataset_name in medmnist.INFO.keys():
                val_y.append(torch.tensor(label[0], dtype=torch.int64))
            else:
                val_y.append(torch.tensor(label))

        val_x = torch.stack(val_x)
        val_y = torch.stack(val_y)
        out_val = [(x,y) for (x,y) in sorted(zip(val_x, val_y), key=lambda v : v[1])]
        val_x,  val_y  = [torch.stack([elem[i] for elem in out_val]) for i in [0,1]]
        val_x, val_y = torch.Tensor(val_x), torch.Tensor(val_y)
        val_idx = (torch.nonzero(val_y[1:] != val_y[:-1], as_tuple=False)[:,0] + 1).tolist()
        val_idx = list(zip([0] + val_idx, val_idx + [None]))

        val_ds = []
        for class_id in range(0, args.n_classes):
            vl_s, vl_e = val_idx[class_id]
            val_ds  += [(val_x[vl_s:vl_e],  val_y[vl_s:vl_e])]

    out_dict = {'train': train_ds,
                'val': val_ds,
                'test': test_ds}
    return out_dict


def split_data_with_assignment(args, cls_assignment=None):
    skip = args.n_classes_per_task
    ds_dict = get_data_per_class(args)

    if cls_assignment == None:
        ds_train = ds_dict['train']
        class_lengths = torch.Tensor([len(ds_class[1]) for ds_class in ds_train])
        sort, cls_assignment = class_lengths.sort(descending=True)
        cls_assignment = cls_assignment.tolist()
        print(sort, cls_assignment)
        
    # for each data split (i.e., train/val/test)
    ds_out = {}
    for name_ds, ds in ds_dict.items():
        split_ds = []
        for i in range(0, args.n_classes, skip):
            t_list = cls_assignment[i:i+skip]
            task_ds_tmp_x = []
            task_ds_tmp_y = []
            for class_id in t_list:
                class_x, class_y = ds[class_id]
                task_ds_tmp_x.append(class_x)
                task_ds_tmp_y.append(class_y)

            task_ds_x = torch.cat(task_ds_tmp_x)
            task_ds_y = torch.cat(task_ds_tmp_y)
            split_ds += [(task_ds_x, task_ds_y)]
        ds_out[name_ds] = split_ds
    
    return ds_out['train'], ds_out['val'], ds_out['test'], cls_assignment


def get_loader_with_assignment(args, cls_assignment=None, run=None):
    if run == None:
        dir_output = f'{args.dir_data}/data_splits/FCL/{args.dataset_name}/{args.overlap}/'
    else:
        dir_output = f'{args.dir_data}/data_splits/FCL/{args.dataset_name}/{args.overlap}/run{run}/'

    loader_fn = f'{dir_output}/{args.dataset_name}_split.pkl'
    cls_assignment_fn = f'{dir_output}/{args.dataset_name}_cls_assignment.pkl'
    if not os.path.exists(loader_fn):
        os.makedirs(dir_output)
        print(cls_assignment)
        train_ds, val_ds, test_ds, cls_assignment = split_data_with_assignment(args, cls_assignment)
        ds_list = [train_ds, val_ds, test_ds]
        loader_list = []
        for ds in ds_list:
            loader_tmp = []
            for task_data in ds:
                images, label = task_data
                indices = torch.from_numpy(np.random.choice(images.size(0), images.size(0), replace=False))
                images = images[indices]
                label = label[indices]
                task_ds = torch.utils.data.TensorDataset(images, label)
                task_loader = torch.utils.data.DataLoader(task_ds, batch_size=args.batch_size, drop_last=True)
                loader_tmp.append(task_loader)
            loader_list.append(loader_tmp)
            
        # save data splits and cls_assignment
        with open(loader_fn, 'wb') as outfile:
            pickle.dump(loader_list, outfile)
            outfile.close()
        with open(cls_assignment_fn, 'wb') as outfile:
            pickle.dump(cls_assignment, outfile)
            outfile.close()

    else:
        loader_list = pickle.load(open(loader_fn, 'rb'))
        cls_assignment = pickle.load(open(cls_assignment_fn, 'rb'))
    
    return loader_list, cls_assignment



def assign_data_per_client(args, run):
    ds_dict = get_data_per_class(args)
    skip = args.n_classes_per_task

    # split datasets in non-overlapping parts
    split_list_x = {}
    split_list_y = {}
    # for each data split (i.e., train/val/test)
    for name_ds, ds in ds_dict.items():
        split_list_x[name_ds] = {}
        split_list_y[name_ds] = {}
        # for each class_id
        for class_id, class_ds in enumerate(ds):
            split_list_x[name_ds][class_id] = {}
            split_list_y[name_ds][class_id] = {}
            class_x, class_y = class_ds
            # split class data in non-overlapping parts of equal size
            split_len = int(len(class_x)/args.n_clients)
            split_x = torch.split(class_x, split_len)
            split_y = torch.split(class_y, split_len)
            # assign the non-overlapping parts to each client
            for client_id in range(args.n_clients):
                split_list_x[name_ds][class_id][client_id] = split_x[client_id]
                split_list_y[name_ds][class_id][client_id] = split_y[client_id]

    if args.dataset_name in medmnist.INFO.keys():
        ds_train = ds_dict['train']
        class_lengths = torch.Tensor([len(ds_class[1]) for ds_class in ds_train])
        sort, cls_assignment = class_lengths.sort(descending=True)
        cls_assignment = cls_assignment.tolist()
        cls_assignment_list = [cls_assignment for client in range(args.n_clients)]
    else:
        cls_assignment_list = []
        for client_id in range(args.n_clients):
            np.random.seed((run+1) * (client_id+1))
            cls_assignment = np.arange(args.n_classes)
            np.random.shuffle(cls_assignment)
            cls_assignment_list.append(cls_assignment)

    # for each client id
    all_clients_ds = []
    for client_id in range(args.n_clients):
        client_ds = []
        class_ids = cls_assignment_list[client_id]
        # for each data split (i.e., train/val/test)
        for name_ds, ds in split_list_x.items():
            client_ds_tmp = []
            # assign non-overlapping split to each client task
            for i in range(0, args.n_classes, skip):
                t_list = class_ids[i:i+skip]
                task_ds_x_tmp = []
                task_ds_y_tmp = []
                for class_id in t_list:
                    task_ds_x_tmp.append(split_list_x[name_ds][class_id][client_id])
                    task_ds_y_tmp.append(split_list_y[name_ds][class_id][client_id])
                
                task_ds_x = torch.cat(task_ds_x_tmp)
                task_ds_y = torch.cat(task_ds_y_tmp)
                # shuffle samples in each task before saving
                perm = torch.randperm(len(task_ds_x))
                task_ds_x, task_ds_y = task_ds_x[perm], task_ds_y[perm]
                client_ds_tmp += [(task_ds_x, task_ds_y)]
            client_ds.append(client_ds_tmp)
        all_clients_ds.append(client_ds)
    
    return all_clients_ds, cls_assignment_list


def get_loader_all_clients(args, run):
    dir_output = f'{args.dir_data}/data_splits/{args.framework}/{args.dataset_name}/{args.overlap}/run{run}/'
    loader_fn = f'{dir_output}{args.dataset_name}_split.pkl'
    cls_assignment_fn = f'{dir_output}{args.dataset_name}_cls_assignment.pkl'
    global_test_fn = f'{dir_output}{args.dataset_name}_global_test.pkl'

    if not os.path.exists(loader_fn):
        os.makedirs(dir_output)
        all_clients_ds, cls_assignment_list = assign_data_per_client(args, run)
        loader_client_list = []
        task_client_list = []
        for client_ds in all_clients_ds:
            loader_list = []
            task_ds_list = []
            # train/val/test splits
            for split_ds in client_ds:
                loader_tmp = []
                task_ds_tmp = []
                for task_data in split_ds:
                    task_ds = torch.utils.data.TensorDataset(task_data[0], task_data[1])
                    task_loader = torch.utils.data.DataLoader(task_ds, batch_size=args.batch_size)
                    loader_tmp.append(task_loader)
                    # we save the TensorDataset to use ConcatDataset and create the global test loader for each task
                    task_ds_tmp.append(task_ds)
                
                loader_list.append(loader_tmp)
                task_ds_list.append(task_ds_tmp)
            loader_client_list.append(loader_list)
            task_client_list.append(task_ds_list)

        global_test_list = []
        for task_id in range(args.n_tasks):
            test_task = []
            for client_id in range(len(loader_client_list)):
                test_task.append(task_client_list[client_id][2][task_id])  # [2] = test set
            
            concat_test_task = torch.utils.data.ConcatDataset(test_task)
            loader_test_task = torch.utils.data.DataLoader(concat_test_task, batch_size=args.batch_size)
            global_test_list.append(loader_test_task)
        
        # save data splits
        with open(loader_fn, 'wb') as outfile:
            pickle.dump(loader_client_list, outfile)
            outfile.close()
        # save class-per-task assignment
        with open(cls_assignment_fn, 'wb') as outfile:
            pickle.dump(cls_assignment_list, outfile)
            outfile.close()
        # save global testing data
        with open(global_test_fn, 'wb') as outfile:
            pickle.dump(global_test_list, outfile)
            outfile.close()

    else:
        loader_client_list = pickle.load(open(loader_fn, 'rb'))
        cls_assignment_list = pickle.load(open(cls_assignment_fn, 'rb'))
        global_test_list = pickle.load(open(global_test_fn, 'rb'))

    return loader_client_list, cls_assignment_list, global_test_list