import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset
from brainscore_vision.model_helpers.activations.temporal.inputs.video import Video

from model import S3D
from training import train

class AFD101(VisionDataset):
    def __init__(self, root, annotation_path, fold=1, train=True):
        super(AFD101, self).__init__(root)
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))
        
        extensions = ('avi',)
        self.fold = fold
        self.train = train

        classes = list(sorted(list_dir(root)))
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}
        self.samples = make_dataset(self.root, self.class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        
        self.indices = self._select_fold(video_list, annotation_path, fold, train)
        self.video_clips = [video_list[i] for i in self.indices]
        self.num_classes = len(classes)

    def _select_fold(self, video_list, annotation_path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(annotation_path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [x[0] for x in data]
            selected_files.extend(data)
        selected_files = set(selected_files)
        # Replace '\\' with '/' for Windows compatibility
        indices = [i for i in range(len(video_list)) if video_list[i][len(self.root) + 1:].replace('\\', '/') in selected_files]
        return indices

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, idx):
        video_path, label = self.samples[self.indices[idx]]
        video = Video.from_path(video_path)
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

    trainset = AFD101(
        root='D:\\AFD101\\videos',
        annotation_path='D:\\AFD101\\splits'
    )
    valset = AFD101(
        root='D:\\AFD101\\videos',
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
        'project': 'S3D-AFD101',
        'arch': 'S3D',
        'dataset': 'AFD101',
        'epochs': args.epochs
    }
    train(model, trainloader, validloader, optim, scheduler, criterion, config, device)