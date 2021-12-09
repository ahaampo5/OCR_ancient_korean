import torch
import numpy as np
import random
from typing import Tuple, Dict, List
import editdistance

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]

def set_seed(seed: int = 21):
    """시드값을 고정하는 함수. 실험 재현을 위해 사용"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_vocab(tokens_paths: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Generation 과정에서 활용할 토큰을 불러와 vocab에 추가하는 함수
    Args:
        tokens_paths (str): 토큰 정보가 담긴 파일 경로(tokens.txt)
    Returns:
        token_to_id: {토큰명:ID} 꼴 딕셔너리
        id_to_token: {ID:토큰명} 꼴 딕셔너리
    """
    tokens = []
    tokens.extend(SPECIAL_TOKENS)
    for tokens_file in tokens_paths:
        with open(tokens_file, "r") as fd:
            reader = fd.read()
            for token in reader.split("\n"):
                if token not in tokens:
                    tokens.append(token)
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i: tok for i, tok in enumerate(tokens)}
    return token_to_id, id_to_token


def id_to_string(tokens, data_loader, do_eval=0):
    """디코더를 통해 얻은 추론 결과를 문자열로 구성된 수식으로 복원하는 함수"""
    result = []
    if do_eval:
        eos_id = data_loader.dataset.token_to_id["<EOS>"]
        special_ids = set(
            [
                data_loader.dataset.token_to_id["<PAD>"],
                data_loader.dataset.token_to_id["<SOS>"],
                eos_id,
            ]
        )

    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != -1:
                        string += data_loader.dataset.id_to_token[token] + " "
                elif token == eos_id:
                    break
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += data_loader.dataset.id_to_token[token] + " "

        result.append(string)
    return result


def word_error_rate(predicted_outputs: list, ground_truths: list) -> float:
    """Word Error Rate(WER)을 계산하는 함수

    Reference.
        - Wikipedia, 'Word error rate', https://en.wikipedia.org/wiki/Word_error_rate
    """
    sum_wer = 0.0
    for output, ground_truth in zip(predicted_outputs, ground_truths):
        output = output.split(" ")
        ground_truth = ground_truth.split(" ")
        distance = editdistance.eval(output, ground_truth)
        length = max(len(output), len(ground_truth))
        sum_wer += distance / length
    return sum_wer / len(predicted_outputs)


def sentence_acc(predicted_outputs: list, ground_truths: list) -> float:
    """Sentence Accuracy를 계산하는 함수

    Sentence Accuracy = { SUM { Indicator { predicted == ground truth } } / Total number of sentences }
    """
    correct_sentences = 0
    for output, ground_truth in zip(predicted_outputs, ground_truths):
        if np.array_equal(output, ground_truth):
            correct_sentences += 1
    return correct_sentences / len(predicted_outputs)