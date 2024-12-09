import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import S3D
from training import train
from datasets import AFD101
    
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
    parser.add_argument('--minlr', type=float, default=1e-6)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset = AFD101(
        root='/mnt/scratch/fkolly/datasets/AFD101/videos',
        annotation_path='/mnt/scratch/fkolly/datasets/AFD101/splits',
        label_start=1000
    )
    valset = AFD101(
        root='/mnt/scratch/fkolly/datasets/AFD101/videos',
        annotation_path='/mnt/scratch/fkolly/datasets/AFD101/splits',
        train=False,
        label_start=1000
    )
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=32, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    validloader = DataLoader(valset, batch_size=4, shuffle=False, num_workers=32, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    model = S3D(num_classes=101)
    model.apply(init_weights) # Weights initialization
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1, end_factor=0.1)
    config = {
        'project': 'S3D-AFD101',
        'arch': 'S3D',
        'dataset': 'AFD101',
        'epochs': args.epochs
    }
    train(model, trainloader, validloader, optim, scheduler, criterion, config, device)