from glob import glob

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .utils import id_to_string, set_seed, load_vocab
from .dataset import MyDataset, get_valid_transforms
from .model import SWIN

token_to_id, id_to_token = load_vocab(['.\\token.txt'])

class CFG:
    seed = 21
    checkpoint_path = '../recognition/pth/model_8_98.18.pth'
    image_paths = '/content/data/크랍이미지/*'
config = CFG()

is_cuda = torch.cuda.is_available()

set_seed(21)

hardware = 'cuda' if is_cuda else 'cpu'
device = torch.device(hardware)
print(device)

transforms = get_valid_transforms()

image_paths = sorted(glob('/content/data/크랍이미지/*')[:30])

test_dataset = MyDataset(
    image_paths, transforms=transforms, mode='test'
)
test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=1,
    num_workers=0,
)

model = SWIN(test_dataset)
checkpoint = torch.load(config.checkpoint_path, map_location='cpu')

model_checkpoint = checkpoint
model.load_state_dict(model_checkpoint)
model.eval()
model.to(device)

