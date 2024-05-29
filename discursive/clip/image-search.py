orig_embeds = '/home/andre/discursive/clip/embeddings/smid_clip_embeddings_orig.jsonl'
# new_embeds = '/home/andre/discursive/clip/embeddings/smid_clip_embeddings_epoch-20_batch-400.jsonl'
new_embeds = '/home/andre/discursive/clip/embeddings/smid_clip_embeddings_v4-e2.jsonl'
model_path = '/home/andre/discursive/clip/model_checkpoints_v4/clip_epoch-2'

IMAGE_FOLDER = '/home/andre/discursive/data/SMID/imgs'

import numpy as np
import json
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


def cosine_sim(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def load_embeds(embeds_path):
    with open(embeds_path, 'r') as f:
        embeds = [json.loads(line) for line in f]
    # turn into dictionary with image as key and features as value
    embeds = {embed['img']: embed['features'][0] for embed in embeds}
    return embeds

orig_embeds = load_embeds(orig_embeds)
new_embeds = load_embeds(new_embeds)

def search_top_k(embedding, embedding_store, k=5):

    # calculate cosine similarity between embedding and all embeddings in embedding_store
    similarities = {img: cosine_sim(embedding, embedding_store[img]) for img in embedding_store}

    # sort by similarity and return top k
    top_k = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[1:k+1]
    return top_k


# load model
from transformers import CLIPProcessor, CLIPModel
import torch
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model = CLIPModel.from_pretrained(model_path).to(DEVICE)

search_query = "that's not fair! | they had it coming"

# get text embedding for search query
inputs = processor(text=search_query, return_tensors='pt', padding='max_length', max_length=77, truncation=True)
inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
text_embed = model.get_text_features(**inputs).detach().cpu().numpy()

top_k_orig = search_top_k(text_embed, orig_embeds, k=5)
top_k_new = search_top_k(text_embed, new_embeds, k=5)

# print path to images
for i, (img, sim) in enumerate(top_k_orig):
    print(f'#{i}: {os.path.join(IMAGE_FOLDER, img)} - similarity: {sim}')

