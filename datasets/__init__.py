import torch
import torchvision
from .random_dataset import RandomDataset
from .objectron_dataset import ObjectronDataset


def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None, only_train=False, objectron_pair="uniform", objectron_exclude=[]):
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'stl10':
        if not only_train:
            dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
        else:
            dataset = torchvision.datasets.STL10(data_dir, split='train' if train else 'test', transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
    elif dataset == 'random':
        dataset = RandomDataset()
    elif dataset == "folder":
        split = "train" #default
        if not train:
            split="valid"
        dataset  = torchvision.datasets.ImageFolder(f"{data_dir}/{split}", transform=transform)
    elif dataset =="objectron":
        split = "train" #default
        if only_train: #For the memory dataloader
            #dataset  = torchvision.datasets.ImageFolder(f"{data_dir}/{split}", transform=transform)
            dataset  = ObjectronDataset(root=data_dir, transform=transform, split="train",
                single=True,objectron_pair=objectron_pair, objectron_exclude=objectron_exclude)
            return dataset
        if not train:
            #split="valid"
            dataset  = ObjectronDataset(root=data_dir, transform=transform, split="valid", 
                single=True,objectron_pair=objectron_pair, objectron_exclude=objectron_exclude)
        else:
            dataset = ObjectronDataset(root=data_dir, transform=transform, split="train", 
                objectron_pair=objectron_pair, objectron_exclude=objectron_exclude)
    else:
        raise NotImplementedError
    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch

    return dataset