import os
import time
from glob import glob
from tqdm import tqdm
import jamotools

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import id_to_string, set_seed
from dataset import MyDataset, get_valid_transforms
from model import SWIN
import jamo

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]

def main(config):
    is_cuda = torch.cuda.is_available()
    
    set_seed(config.seed)

    hardware = 'cuda' if is_cuda else 'cpu'
    device = torch.device(hardware)
    print(f'Running on {device}')

    transforms = get_valid_transforms()

    image_paths = sorted(glob(config.image_paths)) # glob(config.image_paths)

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
    checkpoint = torch.load(config.checkpoint_path)
    model_checkpoint = checkpoint
    model.load_state_dict(model_checkpoint)
    model.eval()
    model.to(device)

    results = []
    with torch.no_grad():
        for image, label in test_loader:
            image = image.float().to(device)

            expected = label.to(device)

            output = model(image, expected, False, teacher_forcing_ratio=0.0)
            decoded_values = output.transpose(1,2)
            _, sequence = torch.topk(decoded_values, 1, dim=1)
            sequence = sequence.squeeze(1) # [B, MAX_LEN]

            sequence_str = id_to_string(sequence, test_loader,do_eval=1)
            
            print(''.join(sequence_str).replace('b_', ''))

            # with open('/content/drive/MyDrive/recognition/test.txt', 'a', encoding='utf-8') as f:
            #     f.write(''.join(sequence_str).replace('b_', '').replace('_', ''))

    
if __name__ == '__main__':
    class CFG:
        seed = 21
        checkpoint_path = './pth/model_8_98.18.pth'
        image_paths = '/content/drive/MyDrive/recognition/test/*'
    config = CFG()
    main(config)