import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import S3D
from training import train
from datasets import AFD101, ImageNetVid
    
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

    imagenettrainset = ImageNetVid(
        root='/mnt/scratch/akgokce/datasets/imagenet',
        split='train'
    )
    imagenetvalset = ImageNetVid(
        root='/mnt/scratch/akgokce/datasets/imagenet',
        split='val'
    )
    afd101trainset = AFD101(
        root='/mnt/scratch/fkolly/datasets/AFD101/videos',
        annotation_path='/mnt/scratch/fkolly/datasets/AFD101/splits',
        label_start=1000
    )
    adf101valset = AFD101(
        root='/mnt/scratch/fkolly/datasets/AFD101/videos',
        annotation_path='/mnt/scratch/fkolly/datasets/AFD101/splits',
        train=False,
        label_start=1000
    )
    # Do we train batch by batch w/ diff. sets or interleave the sets directly? Open question
    concat_train = torch.utils.data.ConcatDataset([imagenettrainset, afd101trainset])
    concat_val = torch.utils.data.ConcatDataset([imagenetvalset, adf101valset])
    train_subset = torch.utils.data.Subset(concat_train, [random.randint(0, len(concat_train)) for _ in range(len(concat_train) // 100)])
    val_subset = torch.utils.data.Subset(concat_val, [random.randint(0, len(concat_val)) for _ in range(len(concat_val) // 100)])
    trainloader = DataLoader(train_subset, batch_size=6, shuffle=True, num_workers=64, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    validloader = DataLoader(val_subset, batch_size=6, shuffle=False, num_workers=64, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    model = S3D(num_classes=1000+101)
    model.apply(init_weights) # Weights initialization
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1, end_factor=0.1)
    config = {
        'project': 'S3D-Mixed',
        'arch': 'S3D',
        'dataset': 'ImageNet+AFD101',
        'epochs': args.epochs
    }
    train(model, trainloader, validloader, optim, scheduler, criterion, config, device, skip_start_log=True)