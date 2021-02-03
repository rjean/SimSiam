import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
import shutil
from tqdm import tqdm
from configs import get_args
from augmentations import get_aug
from models import backbones, get_model
from tools import AverageMeter, PlotLogger, knn_monitor
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import cProfile
import argparse

from augmentations.objectron_aug import get_objectron_gpu_transform
from augmentations.objectron_aug import ObjectronTransform
from datasets.objectron_dataset import ObjectronDataset
from pathlib import Path

import glob
import os




def main():
    parser = argparse.ArgumentParser(description='Command line tool for evaluating zero shot pose estimation on objectron.')
    parser.add_argument("checkpoint", type=str, help="Location of the trained model to use.")
    parser.add_argument("--data_dir", type=str, default="datasets/objectron_96x96", help="Dataset location.")
    parser.add_argument("--image_size", type=int, default=96, help="Crop size.")
    parser.add_argument("--pairs", type=str, default="next", help="Objectron pairing scheme. Do not use this parameter.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of worker processes.")
    parser.add_argument("--model", type=str, default="simsiam", help="Model type to use.")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cuda",help="Device. Defaults to cuda.")
    parser.add_argument("--subset_size", default=1, type=float, help="Training subset size, between 0.01 and 1")
    parser.add_argument("--output_dir", type=str, default="outputs/generate_embeddings")

    
    args = parser.parse_args()
    
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    transform = ObjectronTransform(args.image_size)

    train_dataset  = ObjectronDataset(root=args.data_dir, transform=transform, split="train",
                single=True,objectron_pair=args.pairs, memory=True)

    valid_dataset  = ObjectronDataset(root=args.data_dir, transform=transform, split="valid", 
                single=True,objectron_pair=args.pairs, memory=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    # define model
    model_path = None
    model = get_model(args.model, args.backbone).to(args.device)


    model_path = args.checkpoint
    saved = torch.load(model_path)
    model.load_state_dict(saved['state_dict'])
    model.eval()

    backbone = model.backbone
    #Training embeddings
    generate_embeddings(args, train_loader, train_dataset, backbone, model)
    #Test embeddings
    generate_embeddings(args, valid_loader, valid_dataset, backbone, model, prefix="")
    
    
    
def generate_embeddings(args, dataloader, dataset, backbone, model, prefix="train_"):
    local_progress=tqdm(dataloader)
    
    max_batch = int(args.subset_size*len(dataset)/args.batch_size)
    train_features = torch.zeros((backbone.output_dim, max_batch*args.batch_size)).cuda()
    #train_labels = torch.zeros(max_batch*args.batch_size, dtype=torch.int64).cuda()
    train_targets = []
    train_sequence_uids = []
    with torch.no_grad():
        batch_num = 0
        for idx, (images, labels) in enumerate(local_progress):
            images = images.to(args.device, non_blocking=True)
            meta = labels[1:]
            targets = labels[0]
            base = batch_num*dataloader.batch_size

            with autocast():
                model.zero_grad()
                features = backbone(images)
                features = F.normalize(features, dim=1)
            train_features[:,base:base+len(images)]=features.t()
            #train_labels[base:base+len(images)]=labels
            train_targets += targets.cpu().numpy().tolist()
            train_sequence_uids += meta[0]
            batch_num+=1
            if batch_num >= max_batch:
                break
    
    np.save(f"{args.output_dir}/{prefix}embeddings.npy",train_features.cpu().numpy())
    train_info = np.vstack([np.array(train_targets),np.array(train_sequence_uids)])
    np.save(f"{args.output_dir}/{prefix}info.npy", train_info)


if __name__ == "__main__":
    #args = get_args()
    # print(args.device)
    main()
















