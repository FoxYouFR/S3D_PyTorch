import os
import json
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset

from brainscore_vision.model_helpers.activations.temporal.inputs.video import Video

ListData = namedtuple('ListData', ['id', 'label', 'path'])

def video_from_imgs(imgs, transform):
    # Shape is (T, H, W, C)
    imgs = np.array(imgs)
    # Shape is (T, H, W, C)
    data = torch.from_numpy(imgs).float()
    # Need shape (T, C, H, W) https://discuss.pytorch.org/t/can-transforms-compose-handle-a-batch-of-images/4850/5
    data = data.permute(0, 3, 1, 2)
    data = transform(data)
    # Need shape (C, T, H, W) https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
    data = data.permute(1, 0, 2, 3)
    return data

class SmthSmthV2Dataset(Dataset):
    def __init__(self, root, json_file_input, json_file_labels, is_val, label_start=0, fps=12, duration=2000,
                  size=(224, 224), is_test=False):
        self.json_file_input = json_file_input
        self.json_file_labels = json_file_labels
        self.root = root
        self.fps = fps
        self.duration = duration
        self.size = size
        self.is_val = is_val
        self.is_test = is_test
        self.label_start = label_start

        self.classes = self.read_json_labels()
        self.classes_dict = self.get_two_way_dict(self.classes)
        self.json_data = self.read_json_input()

        self.transform = transforms.Compose([
            transforms.CenterCrop(self.size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


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
        item = self.json_data[index]

        video = Video.from_path(item.path)
        video = video.set_fps(self.fps)
        video = video.set_window(0, self.duration)
        imgs = video.to_frames()

        label = item.label
        label = self.classes_dict[label] + self.label_start

        data = video_from_imgs(imgs, self.transform)
        return data, label
    
    def __len__(self):
        return len(self.json_data)

class ImageNetVid(Dataset):
    def __init__(self, root, label_start=0, fps=12, duration=2000, size=(224, 224), split='train'):
        self.root = os.path.join(root, split)
        self.fps = fps
        self.duration = duration
        self.size = size

        self.image_paths = []
        self.labels = []

        classes = sorted(os.listdir(self.root))
        self.class_to_idx = {class_name: idx+label_start for idx, class_name in enumerate(classes)}

        for class_name in classes:
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(self.class_to_idx[class_name])

        self.transform = transforms.Compose([
            transforms.CenterCrop(self.size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        video = Video.from_img_path(img_path, self.duration, self.fps, size=self.size)
        imgs = video.to_frames()
        data = video_from_imgs(imgs, self.transform)
        return data, label

    def __len__(self):
        return len(self.image_paths)

class AFD101(VisionDataset):
    def __init__(self, root, annotation_path, label_start=0, fold=1, fps=12, duration=2000, size=(224, 224), train=True):
        super(AFD101, self).__init__(root)
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))
        
        extensions = ('avi',)
        self.fold = fold
        self.fps = fps
        self.duration = duration
        self.size = size
        self.train = train

        classes = list(sorted(list_dir(root)))
        self.class_to_idx = {classes[i]: i+label_start for i in range(len(classes))}
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}
        self.samples = make_dataset(self.root, self.class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        
        self.indices = self._select_fold(video_list, annotation_path, fold, train)
        self.video_clips = [video_list[i] for i in self.indices]
        self.num_classes = len(classes)

        self.transform = transforms.Compose([
            transforms.CenterCrop(self.size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
        indices = [i for i in range(len(video_list)) if video_list[i][len(self.root) + 1:] in selected_files]
        return indices

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, idx):
        video_path, label = self.samples[self.indices[idx]]
        video = Video.from_path(video_path)
        video = video.set_fps(self.fps)
        video = video.set_window(0, self.duration)
        imgs = video.to_frames()
        data = video_from_imgs(imgs, self.transform)
        return data, label