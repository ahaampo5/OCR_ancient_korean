import os

import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .utils import load_vocab

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]

class MyDataset(Dataset):
    def __init__(self, image_paths, tokens_path=[os.path.join('.', 'token.txt')], transforms=None, mode='train'):
        self.image_paths = image_paths
        self.token_to_id, self.id_to_token = load_vocab(tokens_path)
        self.transforms = transforms
        self.mode = mode

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image_arr = np.fromfile(path, np.int8)
        image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.mode == 'test':
            dummy_gt = "ᄀ" * 3
            label = torch.Tensor(
                [self.token_to_id[START]] + [self.token_to_id[i] for i in dummy_gt]\
                + [self.token_to_id[END]]).long()
            return image, label

        label = path[-10:-4].split('-')

        label = torch.Tensor(
            [self.token_to_id[START]] + [self.token_to_id[i] for i in label]\
                + [self.token_to_id[END]]).long()

        return image, label

    def __len__(self):
        return len(self.image_paths)
    

def get_train_transforms():
    return A.Compose([
        A.Resize(224,224),
        A.RandomBrightness(limit=0.05),
        A.RandomContrast(limit=0.05),
        A.OpticalDistortion(distort_limit=0.01, shift_limit=0.01),
        A.GaussNoise(var_limit=10.0),
        A.Normalize(),
        ToTensorV2()
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(224,224),
        A.Normalize(),
        ToTensorV2()
    ])