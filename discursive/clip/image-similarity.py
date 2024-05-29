orig_embeds = '/home/andre/discursive/clip/embeddings/smid_clip_embeddings_orig.jsonl'
# new_embeds = '/home/andre/discursive/clip/embeddings/smid_clip_embeddings_epoch-20_batch-400.jsonl'
new_embeds = '/home/andre/discursive/clip/embeddings/smid_clip_embeddings_v6-e9.jsonl'

IMAGE_FOLDER = '/home/andre/discursive/data/SMID/imgs'

import numpy as np
import json
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# clear folder "visuals"
for file in os.listdir('visuals'):
    os.remove(os.path.join('visuals', file))



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

# loop over images and compare top 5 most similar images (show in matplotlib plot saved to folder "visuals")
# plot a 2-column, 5-row grid of the most similar images (do not show the original image)
# first column should be calculated using the original embeddings
# second column should be calculated using the new embeddings
for image in tqdm(orig_embeds):

    file_name = image.split('.')[0]

    top_k_orig = search_top_k(orig_embeds[image], orig_embeds, k=5)
    top_k_new = search_top_k(new_embeds[image], new_embeds, k=5)

    fig, axs = plt.subplots(5, 3, figsize=(10, 15))

    plt.set_cmap('gray')

    for i, (img, sim) in enumerate(top_k_orig):

        image_name = img.split('.')[0]
        img_path = os.path.join(IMAGE_FOLDER, img)
        img = plt.imread(img_path)
        axs[i, 0].imshow(img)
        axs[i, 0].set_title(f'{image_name} - {sim:.2f}')
        # axs[i, 0].axis('off')

        # remove x and y ticks
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])

    for i, (img, sim) in enumerate(top_k_new):
        
        image_name = img.split('.')[0]
        img_path = os.path.join(IMAGE_FOLDER, img)
        img = plt.imread(img_path)
        axs[i, 2].imshow(img)
        axs[i, 2].set_title(f'{image_name} - {sim:.2f}')

        # remove x and y ticks
        axs[i, 2].set_xticks([])
        axs[i, 2].set_yticks([])

    # add "#2", "#3", "#4", "#5", "#6" to the y-axis label of the leftmost column
    for i, ax in enumerate(axs):
        ax[0].set_ylabel(f'#{i+2}', rotation=0, labelpad=20, fontsize=12)

        # turn off axes but keep y label
        # ax[0].axis('off')

    # turn off axes on all third column
    for i in range(5):

        # x ticks y ticks
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])

        if i != 2:
            axs[i, 1].axis('off')

    # add column titles without overwriting other titles
    # add text above the two columns in bold, do not set column because it will overwrite the image titles
    axs[0, 0].text(0, 1.5, 'Set A', fontsize=12, fontweight='bold', transform=axs[0, 0].transAxes)
    axs[0, 2].text(0, 1.5, 'Set B', fontsize=12, fontweight='bold', transform=axs[0, 2].transAxes)

    # add original image on the right side, middle row
    img_path = os.path.join(IMAGE_FOLDER, image)
    img = plt.imread(img_path)
    axs[2, 1].imshow(img)

    axs[2, 1].set_title('Original Image', fontsize=12, fontweight='bold')
    axs[2, 1].axis('off')
    

    plt.savefig(f'visuals/{file_name}.png')
    plt.close()

    
