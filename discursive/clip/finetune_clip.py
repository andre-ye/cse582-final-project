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
import numpy as np

# Constants
BATCH_SIZE = 64
LEARNING_RATE = 5e-6
NUM_EPOCHS = 5
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
VERSION = 'v7'
LOAD_WEIGHTS = 'openai/clip-vit-base-patch32'
# '/home/andre/discursive/clip/model_checkpoints/clip_epoch-20_batch-400'


'''
v4: 5e-6 lr, batch size 64, "A: [comment] B: [reply]" string generation
v5: 5e-6 lr, batch size 64, "[comment]" string generation
v6: 5e-6 lr, batch size 64, "[comment] [reply A] [reply B]
v7: same as v4, but with 0.1 soft negative weight
'''

import torch
import torch.nn.functional as F

def contrastive_loss(image_features, text_features, temperature=0.07):
    """
    Calculates contrastive loss between image and text features.
    Args:
        image_features (torch.Tensor): A tensor of shape (batch_size, dim).
        text_features (torch.Tensor): A tensor of shape (batch_size, dim).
        temperature (float): A scaling factor (temperature) applied to logits.

    Returns:
        torch.Tensor: Scalar tensor containing the loss.
    """
    # Normalize features to unit vectors
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)

    # Calculate logits
    logits = torch.matmul(image_features, text_features.t()) / temperature

    # Create labels
    labels = torch.arange(logits.shape[0]).to(logits.device)

    # Compute cross-entropy loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)

    # Return the average loss
    return (loss_i2t + loss_t2i) / 2

def contrastive_loss_with_soft_negatives(image_features, text_features, temperature=0.07, soft_negative_weight=0.1):
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)

    logits = torch.matmul(image_features, text_features.t()) / temperature
    labels = torch.arange(logits.shape[0]).to(logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)

    # Compute soft negatives
    soft_negatives_i2t = F.cross_entropy(logits, labels, reduction='none')
    soft_negatives_t2i = F.cross_entropy(logits.t(), labels, reduction='none')

    # Add weighted soft negative loss
    loss_i2t += soft_negative_weight * soft_negatives_i2t.mean()
    loss_t2i += soft_negative_weight * soft_negatives_t2i.mean()

    return (loss_i2t + loss_t2i) / 2



def build_string(title, comments, VERSION=VERSION):

    # select comments with at least one reply
    comments = [comment for comment in comments if len(comment['replies']) > 0]

    # if no comments have replies, just make a random selection and make that the string
    if len(comments) == 0:
        comment = comments[random.randint(0, len(comments) - 1)]
        string = ""
        
        if VERSION == 'v4' or VERSION == 'v7':
            string += 'A: "' + comment['body']
            string += '"'
        elif VERSION == 'v5':
            string += comment['body']
        elif VERSION == 'v6':
            string += comment['body']
            # string += ' | ' + 

        return string

    # randomly select a main-level comment
    comment = comments[random.randint(0, len(comments) - 1)]

    # randomly choose two random replies
    replies = comment['replies']

    # random sampling
    # reply_idx_a, reply_idx_b = random.sample(range(len(replies)), 2)

    # deterministic sampling
    reply_idx_a, reply_idx_b = 0, 1

    reply = replies[reply_idx_a]
    reply_b = replies[reply_idx_b]

    # build the string
    string = ""

    # v4
    if VERSION == 'v4' or VERSION == 'v7':
        string += 'A: "' + comment['body']
        string += '"\n'
        string += 'B: "' + reply['body']
        string += '"'
    elif VERSION == 'v5':
        string += comment['body']
    elif VERSION == 'v6':
        string += comment['body']
        string += ' | ' + reply['body']
        string += ' | ' + reply_b['body']

    return string

# Dataset class
class RedditCLIPDataset(Dataset):
    def __init__(self, jsonl_files, processor, idxs=None, version=VERSION):
        self.processor = processor
        self.data = []
        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r') as file:
                for line in file:
                    item = json.loads(line)
                    text = (item['title'], item['comments'])
                    img_url = item['img_url']
                    self.data.append((text, img_url))
        if idxs: self.idxs = idxs
        else: self.idxs = np.arange(len(self.data))
        self.version = version

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        text, img_url = self.data[self.idxs[idx]]
        # text = build_string(text[0], text[1])
        try: text = build_string(text[0], text[1], self.version)
        # allow for keyboard interrupt
        except KeyboardInterrupt: raise
        except: return self[(idx + 1) % len(self)]
        image = load_image(img_url)
        if image is None:
            return self[(idx + 1) % len(self)]
        processed = self.processor(text=text, images=image, return_tensors="pt", padding='max_length', max_length=77, truncation=True)
        # squeeze batch dimension
        processed = {key: val.squeeze(0) for key, val in processed.items()}
        return processed
    


if __name__ == '__main__':

    # Load model and processor
    model = CLIPModel.from_pretrained(LOAD_WEIGHTS).to(DEVICE)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    baseline = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE)


    files = [
        '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_ethical.jsonl',
        '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_intlpolitics.jsonl',
        '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_legal.jsonl',
        '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_social.jsonl',
        '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_affective.jsonl',
    ] # 28571 samples

    dataset = RedditCLIPDataset(files, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # make sure checkpoints folder with correct version exists
    if not os.path.exists(f'model_checkpoints_{VERSION}'):
        os.makedirs(f'model_checkpoints_{VERSION}')

    def cosine_sim(vec_a, vec_b):
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    model.train()
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(dataloader, leave=True)
        batches = 0
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

            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())


            # every 100 batches, compare cosine embeddings of images and save to folder "visuals_training"
            # compare embeddings of 5 images, and save to a matplotlib plot with one row and five columns
            # should show the image and have the cosine similarity between the
            # current trained model and the baseline model as the title
            # save the plot to "visuals_training" folder with the epoch and batch number in the filename

            # if batches % 1 == 0:

            #     # select 5 random images
            #     images = [dataset[random.randint(0, len(dataset) - 1)] for _ in range(5)]

            #     fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            #     for i, processed in enumerate(images):

            #         # get image features from baseline model
            #         with torch.no_grad():
            #             outputs = baseline(**processed)
            #             image_features = outputs.image_embeds

            #         # get image features from current model
            #         outputs = model(**processed)
            #         image_features_new = outputs.image_embeds

            #         # calculate cosine similarity
            #         sim = cosine_sim(image_features.squeeze().cpu().numpy(), image_features_new.squeeze().cpu().numpy())

            #         # plot image
            #         img = processed['images'].squeeze().cpu().numpy().transpose(1, 2, 0)
            #         axs[i].imshow(img)
            #         axs[i].set_title(f'{sim:.2f}')
            #         axs[i].axis('off')

            #     plt.savefig(f'visuals_training/epoch-{epoch+1}_batch-{batches}.png')
            #     plt.close()


            batches += 1

        model.save_pretrained(f'model_checkpoints_{VERSION}/clip_epoch-{epoch+1}')