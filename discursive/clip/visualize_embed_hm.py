'''
Use t-sne to visualize embeddings and color by given feature
'''

# EMBEDDINGS_PATH = '/home/andre/discursive/clip/embeddings/hm_clip_embeddings_orig.jsonl'
EMBEDDINGS_PATH = '/home/andre/discursive/clip/embeddings/hm_clip_embeddings_v7-e5.jsonl'
LABELS = '/home/andre/discursive/data/hatefulmemes/train.jsonl'
# FIGURE_PATH = 'visuals_embed/HM_orig.png'
FIGURE_PATH = 'visuals_embed/HM_v7-e5.png'

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json

# Load embeddings into a list of dictionaries with keys "img", "embed_orig", and "embed_ft"
# make sure to index 0 in embeddings as they are nested in a list
# each embeddings jsonl file contains a key 'img' and 'features'
EMBEDDINGS = {}
with open(EMBEDDINGS_PATH, 'r') as f:
    for line in f:
        data = json.loads(line)
        EMBEDDINGS[data['img']] = data['features'][0]

# read relevant feature
LABELS_MORAL = pd.read_json(LABELS, lines=True)
LABELS_MORAL = LABELS_MORAL[LABELS_MORAL['img'].isin(EMBEDDINGS.keys())]
LABELS_MORAL = LABELS_MORAL[['img', 'label']]

# collect X and coloring
X = np.array([EMBEDDINGS[img] for img in EMBEDDINGS])
coloring = [LABELS_MORAL[LABELS_MORAL['img'] == img]['label'].values[0] for img in EMBEDDINGS]

# run t-sne
X_embedded = TSNE(n_components=2).fit_transform(X)

# visualize and save to figure FIGURE_PATH
plt.figure(figsize=(10, 8), dpi=400)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=coloring, cmap='viridis')
plt.colorbar()
plt.tight_layout()
plt.savefig(FIGURE_PATH)
plt.close()
