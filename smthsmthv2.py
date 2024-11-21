import os
from platform import architecture
import re
import subprocess
import argparse
from collections import namedtuple

import json
import numpy as np
# Required for scikit-video to work correctly
np.float = np.float64
np.int = np.int_
import torch
from torch.utils.data import Dataset, DataLoader
from skvideo.io import FFmpegReader

from model import S3D
from training import train

ListData = namedtuple('ListData', ['id', 'label', 'path'])
ffprobe = lambda file: subprocess.run(
    ['ffprobe', '-v', 'error', '-show_entries', 'format=start_time,duration', file],
    capture_output=True,
    text=True,
    check=True) 

class WebmDataset(Dataset):
    def __init__(self, json_file_input, json_file_labels, root, is_test=False):
        self.json_file_input = json_file_input
        self.json_file_labels = json_file_labels
        self.root = root
        self.is_test = is_test

        self.classes = self.read_json_labels()
        self.classes_dict = self.get_two_way_dict(self.classes)
        self.json_data = self.read_json_input()

    def read_json_labels(self):
        classes = []
        with open(self.json_file_labels, 'rb') as json_file:
            json_reader = json.load(json_file)
            for elem in json_reader:
                classes.append(elem)
        return sorted(classes)
    
    def get_two_way_dict(self, classes):
        classes_dict = {}
        for i, item in enumerate(classes):
            classes_dict[item] = i
            classes_dict[i] = item
        return classes_dict

    def read_json_input(self):
        json_data = []
        if not self.is_test:
            with open(self.json_file_input, 'rb') as json_file:
                json_reader = json.load(json_file)
                for elem in json_reader:
                    label = self.clean_template(elem['template'])
                    if label not in self.classes:
                        raise ValueError(f'Label {label} not found in classes')
                    item = ListData(elem['id'], label, os.path.join(self.root, elem['id'] + '.webm'))
                    json_data.append(item)

        else:
            with open(self.json_file_input, 'rb') as json_file:
                json_reader = json.load(json_file)
                for elem in json_reader:
                    item = ListData(elem['id'], "Dummy label", os.path.join(self.root, elem['id'] + '.webm'))
                    json_data.append(item)

        return json_data
    
    def clean_template(self, template):
        return template.replace('[', '').replace(']', '')
    
    def __getitem__(self, index):
        raise NotImplementedError

class SmthSmthV2Dataset(Dataset):
    def __init__(self, root, json_file_input, json_file_labels, clip_size, nclips, step_size, is_val,
                  is_test=False):
        self.dataset = WebmDataset(json_file_input, json_file_labels, root, is_test=is_test)
        self.json_data = self.dataset.json_data
        self.classes = self.dataset.classes
        self.classes_dict = self.dataset.classes_dict
        self.root = root
        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val
    
    def __getitem__(self, index):
        framerate = 12
        item = self.json_data[index]

        optional_args = {"-r": "%d" % framerate}
        output = ffprobe(item.path)
        start_time, duration = re.findall(r"\d+\.\d+", output.stdout)
        duration = float(duration) - float(start_time)
        if duration is not None:
            nframes = int(duration * framerate)
            optional_args["-vframes"] = "%d" % nframes

        reader = FFmpegReader(item.path, inputdict={}, outputdict=optional_args)

        try:
            imgs = []
            for img in reader.nextFrame():
                imgs.append(img)
        except (RuntimeError, ZeroDivisionError) as e:
            print('{}: WEBM reader cannot open {}. Empty '
                  'list returned.'.format(type(e).__name__, item.path))
            
        num_frames = len(imgs)
        label = item.label
        target_idx = self.classes_dict[label]

        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames
        offset = 0
        if num_frames_necessary < num_frames:
            diff = num_frames - num_frames_necessary
            if not self.is_val:
                offset = np.random.randint(0, diff)

        imgs = imgs[offset:offset + num_frames_necessary : self.step_size]

        if len(imgs) < self.clip_size * self.nclips:
            imgs.extend([imgs[-1]] * (self.clip_size * self.nclips - len(imgs)))

        # Shape is (N, T, H, W, C)
        imgs = np.array(imgs)
        # Shape is (T, H, W, C)
        data = torch.from_numpy(imgs)
        # Need shape (C, H, W, T)
        data = data.permute(3, 1, 2, 0)
        data = data.float() / 255.0
        return (data, target_idx)
    
    def __len__(self):
        return len(self.json_data)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', '--verbose', action='store_true')
    parser.add_argument('-E', '--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset = SmthSmthV2Dataset(
        'D:\\smthsmthv2\\videos',
        'D:\\smthsmthv2\\labels\\train.json',
        'D:\\smthsmthv2\\labels\\labels.json',
        clip_size=36,
        nclips=1,
        step_size=1,
        is_val=False
    )
    valset = SmthSmthV2Dataset(
        'D:\\smthsmthv2\\videos',
        'D:\\smthsmthv2\\labels\\validation.json',
        'D:\\smthsmthv2\\labels\\labels.json',
        clip_size=36,
        nclips=1,
        step_size=1,
        is_val=True
    )
    train_subset = torch.utils.data.Subset(trainset, list(range(0, len(trainset), 500)))
    val_subset = torch.utils.data.Subset(valset, list(range(0, len(valset), 500)))
    trainloader = DataLoader(train_subset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    validloader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    model = S3D(num_classes=174)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    train(model, trainloader, validloader, optim, criterion, args.epochs, args.lr, device)