import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A

from utils import load_vocab

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]

class MyDataset(Dataset):
    def __init__(self, image_paths, tokens_path=['./token.txt'], transforms=None, mode='train'):
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
        
        # if self.mode != 'train':
        #     return image

        label = path[-10:-4].split('-')

        label = torch.Tensor(
            [self.token_to_id[START]] + [self.token_to_id[i] for i in label]\
                + [self.token_to_id[END]]).long()

        return image, label

    def __len__(self):
        return len(self.image_paths)
    

class LoadDataset(Dataset):
    def __init__(
        self,
        groundtruth: str,
        tokens_file: str,
        crop: bool = False,
        preprocessing=True,
        transform=None,
        rgb=3,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        super(LoadDataset, self).__init__()
        self.crop = crop
        self.preprocessing = preprocessing
        self.transform = transform
        self.rgb = rgb
        self.token_to_id, self.id_to_token = load_vocab(tokens_file)
        self.data = [
            {
                "path": p,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        if self.rgb == 3:
            image = image.convert("RGB")
        elif self.rgb == 1:
            image = image.convert("L")
        else:
            raise NotImplementedError

        if self.crop:
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            w, h = image.size
            if h / w > 2:
                image = image.rotate(90, expand=True)
            image = np.array(image)
            image = self.transform(image=image)["image"]

        return {"path": item["path"], "truth": item["truth"], "image": image}
