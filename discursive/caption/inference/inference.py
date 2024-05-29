'''
VQA model inference on the SMID dataset.
'''

# MODEL_PATH = '/home/andre/discursive/caption/checkpoints/model_checkpoints_v4/epoch-13_batch-1500'
MODEL_PATH = '/home/andre/discursive/caption/checkpoints/model_checkpoints_v7/epoch-74'
ACCELERATOR = 'cuda:1'
IMG_DIR = '/home/andre/discursive/data/SMID/imgs'
CAPTION_FILE = '/home/andre/discursive/data/SMID/captions.jsonl'
SAVE_PATH_RAW = '/home/andre/discursive/caption/data/gen_samples/raw/SMID_v7_e74.jsonl'

import torch
import torch.nn.functional as F
import json
import requests
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from transformers import BlipProcessor
from transformers import BlipForQuestionAnswering, AdamW, BlipConfig, BlipTextConfig
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained(MODEL_PATH).to(ACCELERATOR)

def inference(img_path, title):

    # Load the image
    img = Image.open(img_path).convert('RGB')

    # Generate the caption
    inputs = processor(img, title, return_tensors="pt").to(ACCELERATOR)
    out = model.generate(**inputs, max_length=512)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

# make sure file exists, clear it
with open(SAVE_PATH_RAW, 'w') as f:
    f.write('')

# run loop
captions = pd.read_json(CAPTION_FILE, lines=True)
for img in tqdm(captions['img_name']):
    img_path = os.path.join(IMG_DIR, img)
    title = captions[captions['img_name'] == img]['caption'].values[0]
    graph = inference(img_path, title)

    # append json entry
    with open(SAVE_PATH_RAW, 'a') as f:
        json.dump({'img_name': img, 'caption': title, 'graph': graph}, f)
        f.write('\n')