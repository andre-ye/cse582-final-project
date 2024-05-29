'''
Assigns each image a score (0: not included, or 1: included)
Fine-tunes CLIP on the image for one epoch
Uses Bayesian optimization to find the optimal assignment for images
Signal is performance on two downstream tasks (hateful memes and moral predictions)
'''

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from finetune_clip import RedditCLIPDataset, contrastive_loss, contrastive_loss_with_soft_negatives, build_string
import json
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import os
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm
import random
from utils import load_image, graph2text
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
# import standardization
from sklearn.preprocessing import StandardScaler

DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
LOAD_WEIGHTS = 'openai/clip-vit-base-patch32'
LEARNING_RATE = 5e-6

SMID_IMAGE_FOLDER = '/home/andre/discursive/data/SMID/imgs'
HM_IMAGE_FOLDER = '/home/andre/discursive/data/hatefulmemes'
SMID_LABELS = pd.read_csv('/home/andre/discursive/data/SMID/SMID_norms.csv')
HM_LABELS = '/home/andre/discursive/data/hatefulmemes/train.jsonl'
smid_image_names = SMID_LABELS['img_name'].tolist()

files = [
    '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_ethical.jsonl',
    '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_intlpolitics.jsonl',
    '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_legal.jsonl',
    '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_social.jsonl',
    '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_affective.jsonl',
] # 28571 samples

def train_model(idxs):

    # Load model and processor
    model = CLIPModel.from_pretrained(LOAD_WEIGHTS).to(DEVICE)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    baseline = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE)

    dataset = RedditCLIPDataset(files, processor, idxs=idxs, version='v4')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # return model, processor

    model.train()
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        inputs = {key: val.squeeze().to(DEVICE) for key, val in batch.items()}

        # Forward pass
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        # Compute custom contrastive loss
        # loss = contrastive_loss(image_features, text_features)
        loss = contrastive_loss_with_soft_negatives(image_features, text_features)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Training...")
        loop.set_postfix(loss=loss.item())
    
    return model, processor

def eval_model(model, processor):

    '''
    Embed images from SMID and Hateful Memes
    '''

    model.eval()

    SMID_embeds, HM_embeds = {}, {}
    bad = []

    # for file in tqdm(os.listdir(SMID_IMAGE_FOLDER)):
    for file in tqdm(smid_image_names):
    
        # # skip file if not end with .jpg
        # if not file.endswith('.jpg'):
        #     continue

        try:
            img = Image.open(os.path.join(SMID_IMAGE_FOLDER, file + '.jpg'))
            file += '.jpg'
        except:
            bad.append(file)
            continue 

        img = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            img_features = model.get_image_features(**img)
        img_features = img_features.cpu().numpy()
        img_features = img_features.tolist()

        SMID_embeds[file] = img_features

    # for file in tqdm(os.listdir(HM_IMAGE_FOLDER)):
        
    #     img = Image.open(os.path.join(HM_IMAGE_FOLDER, file))
    #     img = processor(images=img, return_tensors="pt").to(DEVICE)
    #     with torch.no_grad():
    #         img_features = model.get_image_features(**img)
    #     img_features = img_features.cpu().numpy()
    #     img_features = img_features.tolist()

    #     HM_embeds[file] = img_features

    # train model on SMID and return train loss
    probe = LinearRegression()
    X = np.array([SMID_embeds[file + '.jpg'][0] for file in SMID_LABELS['img_name'] if file not in bad])
    SMID_LABELS_v2 = SMID_LABELS[~SMID_LABELS['img_name'].isin(bad)]
    y = SMID_LABELS_v2[[col for col in SMID_LABELS.columns if 'mean' in col]]

    # normalize y
    scaler = StandardScaler()
    y = scaler.fit_transform(y)

    probe.fit(X, y)

    # evaluate training loss
    y_pred = probe.predict(X)
    train_loss = np.mean((y - y_pred)**2)

    return train_loss

# Define the objective function
def objective(params):

    # try: 

    idxs_raw = [params[f'sample_{i}'] for i in range(28571)]
    
    # get idxs with 1
    idxs = [i for i, val in enumerate(idxs_raw) if val == 1]

    # train model
    model, processor = train_model(idxs)

    # evaluate model
    train_loss = eval_model(model, processor)

    return {'loss': train_loss, 'status': STATUS_OK}
    
    # except:

    #     return {'loss': 1000, 'status': STATUS_OK}

# Define the search space
space = {f'sample_{i}' : hp.pchoice(f'sample_{i}', [(0.8, 0), (0.2, 1)]) for i in range(28571)}

# Create a Trials object to store the results
trials = Trials()

# Run the optimization
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

# print('Best parameters found:')
# print(best)

# save best to file called best_samples.json
with open('best_samples_50.json', 'w') as f:

    best = {k : int(v) for k, v in best.items()}
    json.dump(best, f)