import torch
import numpy as np

from collections import Counter
from torchvision import transforms
from torchvision.transforms import v2
from utils.data_loader import get_statistics

# modified from https://github.com/optimass/Maximally_Interfered_Retrieval/blob/master/buffer.py
class Memory:
    def __init__(self, args):
        super().__init__()
        self.seen = 0
        self.seen_classes = set()
        self.args = args
        self.memory_x = torch.FloatTensor(args.memory_size, *args.input_size).fill_(0.)
        self.memory_y = torch.LongTensor(args.memory_size).fill_(0.)
        self.memory_t = torch.Tensor(args.memory_size).fill_(0.)

        # define the augmentations for uncertainty-based sampling
        flip = transforms.RandomHorizontalFlip()
        rotation = transforms.RandomRotation(degrees=10) 
        brightness = transforms.ColorJitter(brightness=0.1)
        perspective = transforms.RandomPerspective()
        affine = transforms.RandomAffine(degrees=20, translate=(0.1, 0.3), scale=(0.5, 0.75))
        zoom = transforms.RandomResizedCrop((28, 28), scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True)
        augmentations = torch.nn.Sequential(flip, rotation, brightness, perspective, affine, zoom)
        self.augmentations = augmentations

    @property
    def x(self):
        return self.memory_x

    @property
    def y(self):
        return self.memory_y

    @property
    def t(self):
        return self.memory_t
    

    def reservoir_update(self, samples, labels, task_id):
        for sample, label in zip(samples, labels):
            if self.seen < self.args.memory_size:
                self.x[self.seen] = sample
                self.y[self.seen] = label
                self.t[self.seen] = task_id
            else:
                j = np.random.randint(0, self.seen)
                if j < self.args.memory_size:
                    self.x[j] = sample
                    self.y[j] = label
                    self.t[j] = task_id
            self.seen += 1


    def uncertainty_update(self, samples, labels, task_id, model):
        if self.seen + self.args.batch_size <= self.args.memory_size:   # fill the memory if spots are left
            for sample, label in zip(samples, labels):
                self.x[self.seen] = sample
                self.y[self.seen] = label
                self.t[self.seen] = task_id
                self.seen += 1
        else:                                                           # if full, uncertainty-based update
            samples_tmp = []
            labels_tmp = []
            for sample, label in zip(samples, labels):
                if self.seen < self.args.memory_size:
                    self.x[self.seen] = sample
                    self.y[self.seen] = label
                    self.t[self.seen] = task_id
                    self.seen += 1
                else:
                    samples_tmp.append(sample)
                    labels_tmp.append(label)

            samples = torch.stack(samples_tmp)
            labels = torch.stack(labels_tmp)
            task_ids = torch.Tensor([task_id] * len(labels))           
            # take subsample from the memory
            indices = torch.from_numpy(np.random.choice(self.x.size(0), size=500, replace=False))
            mem_x_tmp, mem_y_tmp, mem_t_tmp = self.x[indices], self.y[indices], self.t[indices]
            # concatenate memory subsample and mini-batch (size: subsample + batch_size)
            mem_candidates_x = torch.cat([mem_x_tmp, samples.detach().cpu()])  
            mem_candidates_y = torch.cat([mem_y_tmp, labels.detach().cpu()])
            mem_candidates_t = torch.cat([mem_t_tmp, task_ids])
            # select data points according to the uncertainty score (size: subsample)
            mem_x_tmp, mem_y_tmp, mem_t_tmp = self.uncertainty_sampling(model, mem_candidates_x,
                                                                        mem_candidates_y, mem_candidates_t,
                                                                        k_value=500,
                                                                        step_str='bottomk')

            # replace old subsample with new one in the memory
            for i, idx in enumerate(indices):
                self.x[idx] = mem_x_tmp[i]
                self.y[idx] = mem_y_tmp[i]
                self.t[idx] = mem_t_tmp[i]


    def class_balanced_update(self, samples, labels, task_id, model, current_classes):
        self.seen_classes.update(labels.unique().cpu().numpy())
        task_ids = torch.Tensor([task_id] * len(labels))
        mem_per_class = self.args.memory_size // len(self.seen_classes)
        
        if self.seen + self.args.batch_size <= self.args.memory_size:   # fill the memory if spots are left
            for sample, label in zip(samples, labels):
                self.x[self.seen] = sample
                self.y[self.seen] = label
                self.t[self.seen] = task_id
                self.seen += 1
        else:                                                           # if full, class-balanced update
            samples_tmp = []
            labels_tmp = []
            for sample, label in zip(samples, labels):
                if self.seen < self.args.memory_size:
                    self.x[self.seen] = sample
                    self.y[self.seen] = label
                    self.t[self.seen] = task_id
                    self.seen += 1
                else:
                    samples_tmp.append(sample)
                    labels_tmp.append(label)

            samples = torch.stack(samples_tmp)
            labels = torch.stack(labels_tmp)
            task_ids = torch.Tensor([task_id] * len(labels))
            mem_tmp_x = []
            mem_tmp_y = []
            mem_tmp_t = []
            mem_candidates_x = torch.cat([self.x, samples.detach().cpu()])  # concatenate current memory and mini-batch
            mem_candidates_y = torch.cat([self.y, labels.detach().cpu()])
            mem_candidates_t = torch.cat([self.t, task_ids])
            mem_per_class_count = sorted(Counter(mem_candidates_y.numpy()).most_common(), key=lambda tup: tup[1])
            mem_used = 0
            count_assigned_classes = 0
            # for each class in the candidate memory set
            for i, (class_id, count) in enumerate(mem_per_class_count):
                class_idx = mem_candidates_y == class_id
                mem_class_x = mem_candidates_x[class_idx]
                mem_class_y = mem_candidates_y[class_idx]
                mem_class_t = mem_candidates_t[class_idx]
                # if a class is under-represented or matches the assigned slots, put all the samples in the memory
                if count <= mem_per_class:
                    mem_tmp_x.append(mem_class_x)
                    mem_tmp_y.append(mem_class_y)
                    mem_tmp_t.append(mem_class_t)
                    count_assigned_classes = i+1
                    mem_used += count
                else:
                    memory_left = self.args.memory_size - mem_used
                    classes_left = len(mem_per_class_count) - count_assigned_classes
                    mem_per_class_tmp = memory_left // classes_left
                    if class_id in current_classes:
                        if mem_class_x.size(0) <= mem_per_class_tmp:
                            mem_tmp_x.append(mem_class_x)
                            mem_tmp_y.append(mem_class_y)
                            mem_tmp_t.append(mem_class_t)
                        else:
                            if self.args.balanced_update == 'random':
                                indices = torch.from_numpy(np.random.choice(mem_class_x.size(0), mem_per_class_tmp, replace=False))
                                mem_class_x_tmp = mem_class_x[indices]
                                mem_class_y_tmp = mem_class_y[indices]
                                mem_class_t_tmp = mem_class_t[indices]
                            if self.args.balanced_update == 'uncertainty':
                                mem_class_x_tmp, mem_class_y_tmp, mem_class_t_tmp = self.uncertainty_sampling(model, mem_class_x,
                                                                                                            mem_class_y, mem_class_t,
                                                                                                            k_value=mem_per_class_tmp,
                                                                                                            step_str=self.args.balanced_step)
                            mem_tmp_x.append(mem_class_x_tmp)
                            mem_tmp_y.append(mem_class_y_tmp)
                            mem_tmp_t.append(mem_class_t_tmp)

                    else:
                        # for the classes already full, reduce the size if needed
                        mem_tmp_x.append(mem_class_x[:mem_per_class_tmp])
                        mem_tmp_y.append(mem_class_y[:mem_per_class_tmp])
                        mem_tmp_t.append(mem_class_t[:mem_per_class_tmp])    

            self.memory_x = torch.cat(mem_tmp_x)
            self.memory_y = torch.cat(mem_tmp_y)
            self.memory_t = torch.cat(mem_tmp_t)
            self.seen += len(samples)


    def random_sampling(self, subsample_size, exclude_task=None):
        if exclude_task is not None:
            valid_indices = (self.t[:self.seen] != exclude_task)
            valid_indices = valid_indices.nonzero().squeeze()
            indices = np.random.choice(valid_indices, subsample_size, replace=False)
            return self.x[indices], self.y[indices], self.t[indices]
        if self.x.size(0) < subsample_size:
            return self.x, self.y, self.t
        else:
            indices = torch.from_numpy(np.random.choice(self.x.size(0), subsample_size, replace=False))
            return self.x[indices], self.y[indices], self.t[indices]
        

    def balanced_random_sampling(self, subsample_size, exclude_task=None):  
        if exclude_task is not None:
            valid_indices = (self.t[:self.seen] != exclude_task)
            valid_indices = valid_indices.nonzero().squeeze()
            classes = self.y[valid_indices].unique()
            subsample_size_per_class = subsample_size // classes.size(0)
            leftover_subsamples = subsample_size % classes.size(0)

            indices = np.array([])
            for cls in classes:
                class_valid_indices = ((self.y[valid_indices] == cls).nonzero().squeeze())
                if leftover_subsamples > 0:
                    subsample_sum = subsample_size_per_class + 1
                else:
                    subsample_sum = subsample_size_per_class
                leftover_subsamples -= 1
                indices = np.concatenate((indices, np.random.choice(class_valid_indices, subsample_sum, replace=False)), None)
            return self.x[indices], self.y[indices], self.t[indices]
        
        else:
            if self.x.size(0) < subsample_size:
                return self.x, self.y, self.t
            
            else:
                classes = self.y.unique()
                subsample_size_per_class = subsample_size // classes.size(0)
                leftover_subsamples = subsample_size % classes.size(0)
                indices = np.array([])
                for cls in classes:
                    class_indices = ((self.y == cls).nonzero().squeeze())
                    if leftover_subsamples > 0:
                        subsample_sum = subsample_size_per_class + 1
                    else:
                        subsample_sum = subsample_size_per_class
                    leftover_subsamples -= 1

                    if class_indices.size(0) < subsample_sum:
                        indices = np.concatenate((indices,class_indices), None)
                    else:
                        indices = np.concatenate((indices, np.random.choice(class_indices, subsample_sum, replace=False)), None)
                return self.x[indices], self.y[indices], self.t[indices]
    

    def uncertainty_sampling(self, model, mem_x=None, mem_y=None, mem_t=None,
                             subsample_size=50, exclude_task=None, k_value=None, step_str=None):
        if mem_x == None:
            # select a subsample (subsample_size) from the whole memory
            mem_x, mem_y, mem_t = self.random_sampling(subsample_size, exclude_task)
        if k_value == None:
            # set the sample size equal to the batch size
            k_value = self.args.batch_size
        if step_str == None:
            step_str = self.args.step_str

        # compute uncertainty scores for the given subsample
        unc_scores, descending_flag = compute_uncertainty_scores(self.args, mem_x, model, self.augmentations, seen_cls=self.seen_classes)
        # extract the samples based on the uncertainty score
        # we assume to sample a number of samples equal to the batch size
        if step_str == 'step':      # step-sized sampling
            skip = mem_x.size(0) // k_value
            steps = np.arange(0, mem_x.size(0), skip)
            score_idx = torch.sort(unc_scores, descending=True)[1][steps]
        if step_str == 'topk':      # top-k sampling
            score_idx = torch.sort(unc_scores, descending=descending_flag)[1][:k_value]
        if step_str == 'bottomk':   # bottom-k sampling
            descending_flag = not descending_flag
            score_idx = torch.sort(unc_scores, descending=descending_flag)[1][:k_value]

        x, y, t = mem_x[score_idx], mem_y[score_idx], mem_t[score_idx]
        return x, y, t


def compute_uncertainty_scores(args, mem_x, model, augmentations, tta_rep=5, seen_cls=None):
    # evaluate prediction on the augmented images given the sequence of transform functions
    # and store the corresponding logits
    transform_cands = [
        CutoutAfterToTensor(args, 1, 10),
        CutoutAfterToTensor(args, 1, 20),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(degrees=10),
        v2.RandomRotation(45),
        v2.RandomRotation(90),
        v2.ColorJitter(brightness=0.1),
        v2.RandomPerspective(),
        v2.RandomAffine(degrees=20, translate=(0.1, 0.3), scale=(0.5, 0.75)),
        v2.RandomResizedCrop(args.input_size[1:], scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomInvert()
            ]

    transformSize = len(transform_cands)
    mem_x = mem_x.to(args.device)
    # all_logits = []
    # with torch.no_grad():
    #     for rep in range(tta_rep):
    #         bx_tmp = augmentations(mem_x)
    #         logits_tmp = model(bx_tmp)
    #         all_logits.append(logits_tmp)

    all_logits = []
    with torch.no_grad():
        for tr in transform_cands:
            bx_tmp = tr(mem_x)
            logits_tmp = model(bx_tmp)
            all_logits.append(logits_tmp)

    # compute uncertainty scores for the current batch extract the indices of the 10 most (un)certain samples
    # we assume to sample a number of samples equal to the batch size
    logits_out = torch.stack(all_logits).detach().cpu()

    if args.uncertainty_score == "bregman":
        unc_scores, descending = BI_LSE(logits_out)
    if args.uncertainty_score == "confidence":
        unc_scores, descending = leastConfidence(logits_out)
    if args.uncertainty_score == "margin":
        unc_scores, descending = marginSampling(logits_out)
    if args.uncertainty_score == "entropy":
        unc_scores, descending = entropy(logits_out)
    if args.uncertainty_score == "rainbow":
        unc_scores, descending = rainbowSampling(logits_out, args, size=transformSize) # size=tta_rep
    elif args.uncertainty_score == "ratio":
        unc_scores, descending = ratioSampling(logits_out)
    return unc_scores, descending


# taken from https://github.com/MLO-lab/Uncertainty_Estimates_via_BVD
def BI_LSE(zs, axis=0, class_axis=-1):
    '''
    Bregman Information of random variable Z generated by G = LSE
    BI_G [ Z ] = E[ G( Z ) ] - G( E[ Z ] )
    We estimate with dataset zs = [Z_1, ..., Z_n] via
    1/n sum_i G( Z_i ) - G( 1/n sum_i Z_i )
    
    Arg zs: Tensor with shape length >= 2
    Arg axis: Axis of the samples to average over
    Arg class_axis: Axis of the class logits
    Output: Tensor with shape length reduced by two
    '''
    E_of_LSE = zs.logsumexp(axis=class_axis).mean(axis)
    LSE_of_E = zs.mean(axis).unsqueeze(axis).logsumexp(axis=class_axis).squeeze(axis)
    bi_scores = E_of_LSE - LSE_of_E
    return bi_scores, True


def leastConfidence(zs, axis=0, class_axis=-1):
    confidence_score = 1 - zs.softmax(class_axis).mean(axis).max(class_axis)[0]
    return confidence_score, True


def marginSampling(zs, axis=0, class_axis=-1):
    softmax_scores = zs.softmax(class_axis).mean(axis)
    top_candidates = torch.topk(softmax_scores, k=2, dim=class_axis)[0]
    firstConfidence = top_candidates[:, 0]
    secondConfidence = top_candidates[:, 1]
    margin_score = firstConfidence - secondConfidence
    return margin_score, False


def entropy(zs, axis=0, class_axis=-1):
    softmax_scores = zs.softmax(class_axis).mean(axis)
    entropy_score = -((softmax_scores * softmax_scores.log()).sum(axis=class_axis))
    return entropy_score, True


def rainbowSampling(zs, args, size=5, axis=0, class_axis=-1):
    counter = torch.zeros(zs.shape)
    top_classes = torch.argmax(zs, class_axis)[:, :, None]
    m = counter.scatter(class_axis, top_classes, 1.0).sum(axis).max(1)[0]
    agreement_score = 1 - m / size
    return agreement_score, True


def ratioSampling(zs, axis=0, class_axis=-1):
    softmax_scores = torch.nn.functional.softmax(zs, class_axis).mean(axis)
    top_candidates = torch.topk(softmax_scores, k=2, dim=class_axis)[0]
    firstConfidence = top_candidates[:, 0]
    secoundConfidence = top_candidates[:, 1]
    margin = secoundConfidence / firstConfidence
    return margin, True


class CutoutAfterToTensor(object):
    '''
    https://davidstutz.de/2-percent-test-error-on-cifar10-using-pytorch-autoagument/
    Note that the fill_color will, on CIFAR-10, be a 3-tuple of average RGB values (i.e., per channel the mean value across the training set is used). As mentioned in the name, i.e., CutoutAfterToTensor, the code is to be used as a transform after applying torchvision.transforms.ToTensor.
    '''
    def __init__(self, args,  n_holes, length, fill_color=torch.tensor([0,0,0])):
        self.n_holes = n_holes
        self.length = length
        self.fill_color = fill_color
        self.args = args
        # for this case:
        mean, std, n_classes, inp_size, in_channels = get_statistics(self.args)
        self.fill_color = torch.Tensor(mean).to(device=self.args.device)


    def __call__(self, img):
        h = img.shape[2]
        w = img.shape[3]
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask).to(device=self.args.device)
        mask = mask.expand_as(img)
        img = img * mask + (1 - mask) * self.fill_color[:, None, None]
        return img


class Solarize:
    def __init__(self, args, v):
        assert 0 <= v <= 1
        self.v = v
        self.args = args

    def __call__(self, pil_img):
        mean, std, n_classes, inp_size, in_channels = get_statistics(self.args)
        invTrans = transforms.Compose([ transforms.Normalize(mean = np.dot(0, mean),
                                                            std = np.divide(1, std)),
                                        transforms.Normalize(mean = np.dot(-1, mean),
                                                            std = np.divide(std, std)),
                                    ])
        trans = transforms.Normalize(mean, std)
        pil_img = invTrans(pil_img)
        mask = pil_img > self.v
        pil_img[mask] = 1 - pil_img[mask]
        return trans(pil_img)