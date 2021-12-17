from flask import Flask, request
from PIL import Image
import base64
import json
import io
# from app.inference import inference
from flask_cors import CORS, cross_origin
import numpy as np
from glob import glob

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import id_to_string, set_seed, load_vocab
from dataset import MyDataset, get_valid_transforms
from model import SWIN

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]

def inference(image): # h, w, c
    image = get_valid_transforms()(image=image)['image']

    dummy_gt = "ᄀ" * 3

    label = torch.Tensor(data=
        [token_to_id[START]] + [token_to_id[i] for i in dummy_gt] + [token_to_id[END]]).long()
    
    image = image.float().to(device)
    image = image.unsqueeze(0)
    expected = label.to(device)
    expected = expected.unsqueeze(0)
    
    output = model(image, expected, False, teacher_forcing_ratio=0.0)
    decoded_values = output.transpose(1,2)
    _, sequence = torch.topk(decoded_values, 1, dim=1)
    sequence = sequence.squeeze(1) # [B, MAX_LEN]

    sequence_str = id_to_string(sequence, token_to_id, id_to_token, do_eval=1)

    return sequence_str

class CFG:
    seed = 21
    checkpoint_path = '../recognition/pth/model_8_98.18.pth'
    image_paths = '/content/data/크랍이미지/*'

config = CFG()
token_to_id, id_to_token = load_vocab(['.\\token.txt'])

is_cuda = torch.cuda.is_available()
set_seed(21)

hardware = 'cuda' if is_cuda else 'cpu'
device = torch.device(hardware)

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

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)

@app.route("/recognize", methods=["GET", "POST"])
@cross_origin()
def susik_recognize():
    if request.method == 'POST':
        image = request.json['image']
        
        image = image.split(",")[1]
        image = base64.b64decode(image)
        image = io.BytesIO(image)
        image = Image.open(image)
        
        data=inference(image)
        data = {'result':data}
        return json.dumps(data)
    else:
        return "Hello World!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6006, debug=True)
