import os
import time
from glob import glob
from tqdm import tqdm
import yaml
import shutil
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from utils import set_seed, id_to_string, word_error_rate, sentence_acc
from dataset import MyDataset
from model import SWIN
from scheduler import (
    CustomCosineAnnealingWarmUpRestarts,
    TeacherForcingScheduler
)

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]

def _train_one_epoch(
    data_loader,
    model,
    criterion,
    optimizer,
    lr_scheduler,
    max_grad_norm,
    device,
    tf_scheduler,
):
    torch.set_grad_enabled(True)
    model.train()

    losses = []
    grad_norms = []
    correct_symbols = 0
    total_symbols = 0
    wer = 0
    num_wer = 0
    sent_acc = 0
    num_sent_acc = 0

    with tqdm(
        total=len(data_loader.dataset),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for image, expected in data_loader:
            image = image.to(device).float()
            tf_ratio = tf_scheduler.step()  # Teacher Forcing Scheduler
            curr_batch_size = len(image)
            expected = expected.to(device)
            expected[expected == -1] = data_loader.dataset.token_to_id[PAD]

            output = model(image, expected, True, tf_ratio)  # [B, MAX_LEN, VOCAB_SIZE]

            decoded_values = output.transpose(1, 2)  # [B, VOCAB_SIZE, MAX_LEN]
            _, sequence = torch.topk(decoded_values, k=1, dim=1)  # [B, 1, MAX_LEN]
            sequence = sequence.squeeze(1)  # [B, MAX_LEN]

            loss = criterion(decoded_values, expected[:, 1:])  # [SOS] 이후부터
            optim_params = [
                p
                for param_group in optimizer.param_groups
                for p in param_group["params"]
            ]
            optimizer.zero_grad()
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(optim_params, max_norm=max_grad_norm)
            grad_norms.append(grad_norm)

            optimizer.step()
            losses.append(loss.item())

            expected[expected == data_loader.dataset.token_to_id[PAD]] = -1
            expected_str = id_to_string(expected, data_loader, do_eval=1)
            sequence_str = id_to_string(sequence, data_loader, do_eval=1)
            wer += word_error_rate(sequence_str, expected_str)
            num_wer += 1
            sent_acc += sentence_acc(sequence_str, expected_str)
            num_sent_acc += 1
            correct_symbols += torch.sum(sequence == expected[:, 1:], dim=(0, 1)).item()
            total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

            pbar.update(curr_batch_size)
            lr_scheduler.step()


    expected = id_to_string(expected, data_loader)
    sequence = id_to_string(sequence, data_loader)

    result = {
        "loss": np.mean(losses),
        "correct_symbols": correct_symbols,
        "total_symbols": total_symbols,
        "wer": wer,
        "num_wer": num_wer,
        "sent_acc": sent_acc,
        "num_sent_acc": num_sent_acc,
    }

    try:
        result["grad_norm"] = np.mean([tensor.cpu() for tensor in grad_norms])
    except:
        result["grad_norm"] = np.mean(grad_norms)

    return result

def _valid_one_epoch(data_loader, model, criterion, device):
    model.eval()

    losses = []
    correct_symbols = 0
    total_symbols = 0
    wer = 0
    num_wer = 0
    sent_acc = 0
    num_sent_acc = 0

    NO_TEACHER_FORCING = 0.0

    with torch.no_grad():
        with tqdm(
            total=len(data_loader.dataset),
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            for image, expected in data_loader:
                image = image.to(device).float()

                curr_batch_size = len(image)
                expected = expected.to(device)

                expected[expected == -1] = data_loader.dataset.token_to_id[PAD]
                output = model(image, expected, False, NO_TEACHER_FORCING)

                decoded_values = output.transpose(1, 2)  # [B, VOCAB_SIZE, MAX_LEN]
                _, sequence = torch.topk(
                    decoded_values, 1, dim=1
                )  # sequence: [B, 1, MAX_LEN]
                sequence = sequence.squeeze(1)  # [B, MAX_LEN], 각 샘플에 대해 시퀀스가 생성 상태

                loss = criterion(decoded_values, expected[:, 1:])
                losses.append(loss.item())

                expected[expected == data_loader.dataset.token_to_id[PAD]] = -1
                expected_str = id_to_string(expected, data_loader, do_eval=1)
                sequence_str = id_to_string(sequence, data_loader, do_eval=1)
                wer += word_error_rate(sequence_str, expected_str)
                num_wer += 1
                sent_acc += sentence_acc(sequence_str, expected_str)
                num_sent_acc += 1
                correct_symbols += torch.sum(
                    sequence == expected[:, 1:], dim=(0, 1)
                ).item()
                total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

                pbar.update(curr_batch_size)

    expected = id_to_string(expected, data_loader)
    sequence = id_to_string(sequence, data_loader)

    result = {
        "loss": np.mean(losses),
        "correct_symbols": correct_symbols,
        "total_symbols": total_symbols,
        "wer": wer,
        "num_wer": num_wer,
        "sent_acc": sent_acc,
        "num_sent_acc": num_sent_acc,
    }
    return result


def main(config):
    set_seed(21)

    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print('------------------------------------')
    print(f"Running on device {device}")

    image_paths = glob('/content/data/크랍이미지/*')

    image_paths = image_paths[:len(image_paths)//10]

    train_transforms = A.Compose([
        A.Resize(224,224),
        A.Normalize(),
        ToTensorV2()
    ])

    valid_transforms = A.Compose([
        A.Resize(224,224),
        A.Normalize(),
        ToTensorV2()
    ])

    train_dataset = MyDataset(image_paths, transforms=train_transforms, mode='train')
    valid_dataset = MyDataset(image_paths, transforms=valid_transforms, mode='valid')
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    for i, j in train_loader :
        print(i, j)
        break
    model = SWIN(train_dataset)
    model.to(device)
    model.train()
    

    criterion = model.criterion.to(device)

    enc_params_to_optimise = [
        param for param in model.encoder.parameters() if param.requires_grad
    ]

    dec_params_to_optimise = [
        param for param in model.decoder.parameters() if param.requires_grad
    ]

    params_to_optimise = [*enc_params_to_optimise, *dec_params_to_optimise]

    optimizer = optim.Adam(params_to_optimise, lr=0, weight_decay=0.00001)

    total_steps = len(train_loader) * config.epochs
    t_0 = total_steps // 1
    t_up = int(t_0 * 0.1) # t_o * warm up ratio
    lr_scheduler = CustomCosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=t_0,
        T_mult=1,
        eta_max=0.0001,
        T_up=t_up,
        gamma=0.8,
    )
    
    tf_scheduler = TeacherForcingScheduler(
        num_steps=total_steps,
        tf_max=0.8,
        tf_min=0.4
    )

    best_score = 0.0
    
    print('Start Train!')
    for epoch in range(config.epochs):
        start_time = time.time()

        train_result = _train_one_epoch(
            data_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            max_grad_norm=1.0,
            device=device,
            tf_scheduler=tf_scheduler,
        )

        train_epoch_sentence_accuracy = (
            train_result["sent_acc"] / train_result["num_sent_acc"]
        )

        train_epoch_wer = train_result["wer"] / train_result["num_wer"]

        train_epoch_score = 0.9*train_epoch_sentence_accuracy + \
            0.1*(1-train_epoch_wer)

        valid_result = _valid_one_epoch(
            data_loader=valid_loader,
            model=model,
            criterion=criterion,
            device=device,
        )

        valid_epoch_sentence_accuracy = (
            valid_result["sent_acc"] / valid_result["num_sent_acc"]
        )

        valid_epoch_wer = valid_result["wer"] / valid_result["num_wer"]

        valid_epoch_score = 0.9*valid_epoch_sentence_accuracy + \
            0.1*(1-valid_epoch_wer)

        if best_score < 0.9*valid_epoch_sentence_accuracy + \
                    0.1*(1-valid_epoch_wer):
            torch.save(model.state_dict(), '/content/drive/Othercomputers/내 컴퓨터/workspace/ocr_ancient_korean/recognition/pth/model.pth')
            best_score = 0.9*valid_epoch_sentence_accuracy + \
                    0.1*(1-valid_epoch_wer)
            print(f'Best score : {best_score}')
            print('model saved')

        
if __name__ == '__main__':
    class CFG:
        epochs = 10
        batch_size = 128
    config = CFG()
    main(config)