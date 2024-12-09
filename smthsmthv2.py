import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import S3D
from training import train
from datasets import SmthSmthV2Dataset

def init_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', '--verbose', action='store_true')
    parser.add_argument('-E', '--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset = SmthSmthV2Dataset(
       '/mnt/scratch/fkolly/datasets/smthsmthv2/videos', # 'D:\\smthsmthv2\\videos',
        '/mnt/scratch/fkolly/datasets/smthsmthv2/labels/train.json', # 'D:\\smthsmthv2\\labels\\train.json',
        '/mnt/scratch/fkolly/datasets/smthsmthv2/labels/labels.json', # 'D:\\smthsmthv2\\labels\\labels.json',
        is_val=False
    )
    valset = SmthSmthV2Dataset(
        '/mnt/scratch/fkolly/datasets/smthsmthv2/videos', # 'D:\\smthsmthv2\\videos',
        '/mnt/scratch/fkolly/datasets/smthsmthv2/labels/validation.json', # 'D:\\smthsmthv2\\labels\\validation.json',
        '/mnt/scratch/fkolly/datasets/smthsmthv2/labels/labels.json', # 'D:\\smthsmthv2\\labels\\labels.json',
        is_val=True
    )
    # Do we train batch by batch w/ diff. sets or interleave the sets directly? Open question
    train_subset = torch.utils.data.Subset(trainset, list(range(0, len(trainset), 100)))
    val_subset = torch.utils.data.Subset(valset, list(range(0, len(valset), 100)))
    trainloader = DataLoader(
        train_subset, batch_size=4, shuffle=True, num_workers=32, pin_memory=True, prefetch_factor=4, persistent_workers=True
    )
    validloader = DataLoader(
        val_subset, batch_size=4, shuffle=False, num_workers=32, pin_memory=True, prefetch_factor=4, persistent_workers=True
    )
    model = S3D(num_classes=174)
    model.apply(init_weights) # Weights initialization
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1, end_factor=0.1)
    config = {
        'project': 'S3D-smthsmthv2',
        'arch': 'S3D',
        'dataset': 'smthsmthv2',
        'epochs': args.epochs
    }
    train(model, trainloader, validloader, optim, scheduler, criterion, config, device)