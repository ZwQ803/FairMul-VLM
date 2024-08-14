import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import Custom_dataset
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.utils.data import Sampler
import random


def read_dataset(dataset_name, model_str, setting='default', config_path='config.json', test=False):
    train_data = Custom_dataset(dataset_name, model_str, setting, config_path, mode = 'train')
    test_data = Custom_dataset(dataset_name, model_str, setting, config_path,mode = 'test')
    val_data = Custom_dataset(dataset_name, model_str, setting, config_path,mode = 'val')
    if test:
        test_loader = DataLoader(test_data, batch_size=32, sampler = SequentialSampler(test_data), num_workers=10)
        return None, test_loader
    weights_train = make_weights_for_balanced_classes_split(train_data)
    train_loader = DataLoader(train_data, batch_size=128, sampler = WeightedRandomSampler(weights_train, len(weights_train)), num_workers=10)
    val_loader = DataLoader(val_data, batch_size=128, sampler = SequentialSampler(val_data), num_workers=10)
    test_loader = DataLoader(test_data, batch_size=128, sampler=SequentialSampler(test_data), num_workers=10)

    return train_loader, test_loader, val_loader

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N/len(dataset.img_cls_ids[c]) for c in range(len(dataset.img_cls_ids))]
    weight = [0] * int(N)
    for idx in tqdm(range(len(dataset))):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam