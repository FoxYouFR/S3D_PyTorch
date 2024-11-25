import os
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from brainscore_vision.model_helpers.activations.temporal.inputs.video import Video

from model import S3D
from training import train

class ImageNet(Dataset):
    def __init__(self, root, annotation_path, train=True):
        super(ImageNet, self).__init__(root)
        

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, idx):
        img_path = None
        label = None
        video = Video.from_img_path(img_path)
        imgs = video.to_frames()
        # TODO batch size?
        # Shape is (T, H, W, C)
        imgs = np.array(imgs)
        data = torch.from_numpy(imgs)
        # Need shape (C, H, W, T)
        data = data.permute(3, 1, 2, 0)
        data = data.float() / 255.0
        return data, label
    
def init_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', '--verbose', action='store_true')
    parser.add_argument('-E', '--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--minlr', type=float, default=1e-6)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset = ImageNet(
        root='/mnt/scratch/ytang/imagenet',
        annotation_path='D:\\AFD101\\splits'
    )
    valset = ImageNet(
        root='/mnt/scratch/ytang/imagenet',
        annotation_path='D:\\AFD101\\splits',
        train=False
    )
    # Do we train batch by batch w/ diff. sets or interleave the sets directly? Open question
    # train_subset = torch.utils.data.Subset(trainset, list(range(0, len(trainset), 500)))
    # val_subset = torch.utils.data.Subset(valset, list(range(0, len(valset), 500)))
    # TODO Find a way to make batch size higher than 1
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    validloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    model = S3D(num_classes=101)
    model.apply(init_weights) # Weights initialization
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=args.minlr)
    # TODO Maybe add warmup?
    config = {
        'project': 'S3D-ImageNet',
        'arch': 'S3D',
        'dataset': 'ImageNet',
        'epochs': args.epochs
    }
    train(model, trainloader, validloader, optim, scheduler, criterion, config, device)