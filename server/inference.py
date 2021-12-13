from . import model, token_to_id, id_to_token, device
from utils import id_to_string
from dataset import get_valid_transforms
import torch

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]

def inference(image): # h, w, c
    image = get_valid_transforms()(image=image)['image']

    dummy_gt = "ᄀ" * 3
    label = torch.Tensor(
        [token_to_id[START]] + [[token_to_id[i]] for i in dummy_gt]\
            + [token_to_id[END]]).long()
    
    image = image.float().to(device)
    
    expected = label.to(device)

    output = model(image, expected, False, teacher_forcing_ratio=0.0)
    decoded_values = output.transpose(1,2)
    _, sequence = torch.topk(decoded_values, 1, dim=1)
    sequence = sequence.squeeze(1) # [B, MAX_LEN]

    sequence_str = id_to_string(sequence, token_to_id, id_to_token, do_eval=1)

    return sequence_str