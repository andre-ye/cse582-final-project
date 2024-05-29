import torch
import torch.nn.functional as F
import json
import requests
from PIL import Image
import pandas as pd
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
import torch
import os
import torch.optim as optim
from tqdm import tqdm
import random

'''
parameters
'''

DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# LOAD_WEIGHTS = 'openai/clip-vit-base-patch32'
LOAD_WEIGHTS = '/home/andre/discursive/clip/model_checkpoints_v7/clip_epoch-5'

# SAVE_PATH = 'embeddings/hm_clip_embeddings_orig.jsonl'
SAVE_PATH = 'embeddings/hm_clip_embeddings_v7-e5.jsonl'

IMAGE_FOLDER = '/home/andre/discursive/data/hatefulmemes'
imgs = pd.read_json('/home/andre/discursive/data/hatefulmemes/train.jsonl', lines=True)['img'].tolist()


'''
run model on images
'''

# create / rewrite new file
with open(SAVE_PATH, 'w') as f:
    f.write('')

processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model = CLIPModel.from_pretrained(LOAD_WEIGHTS).to(DEVICE)

model.eval()

for file in tqdm(imgs):

    img = Image.open(os.path.join(IMAGE_FOLDER, file))
    img = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        img_features = model.get_image_features(**img)
    img_features = img_features.cpu().numpy()
    img_features = img_features.tolist()

    with open(SAVE_PATH, 'a') as f:
        json.dump({'img': file, 'features': img_features}, f)
        f.write('\n')
