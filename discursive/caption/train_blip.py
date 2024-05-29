import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import requests
from transformers import BlipProcessor
from utils import graph2text, load_image
import numpy as np

NUM_EPOCHS = 100
MAX_SEQ_LEN = 512
BATCH_SIZE = 8
LEARNING_RATE = 5e-6
# sess1: 5e-6
# sess2: 1e-5
ACCELERATOR = 'cuda:1'
EDITION = 'v8'
'''
v6: batch size 4, learning rate 1e-6
v7: continued learning, overfitting maybe?
v8: from scratch again (put on pause)
'''

FROM_PRETRAINED = "Salesforce/blip-image-captioning-base" # "Salesforce/blip-itm-large-flickr"
LOAD_WEIGHTS = "Salesforce/blip-image-captioning-base"
# LOAD_WEIGHTS = "/home/andre/discursive/caption/checkpoints/model_checkpoints_v4/epoch-13_batch-1500"
# LOAD_WEIGHTS = "Salesforce/blip-image-captioning-large"
# LOAD_WEIGHTS = "/home/andre/discursive/caption/checkpoints/model_checkpoints_v5/epoch-8"
# LOAD_WEIGHTS = '/home/andre/discursive/caption/checkpoints/model_checkpoints_v6/epoch-3'

# make model checkpoints if not existed
import os
if not os.path.exists(f"checkpoints/model_checkpoints_{EDITION}"):
    os.makedirs(f"checkpoints/model_checkpoints_{EDITION}")


class VQADataset(Dataset):
    def __init__(self, data_files, processor):
        self.data = []
        self.index = []
        self.processor = processor

        temp_ids = set()

        for packet in data_files:
            data_file = packet['file']
            weight = packet['weight']
            # if is weight is 1.2, choose 1 with 20% probability and 2 with 80% probability
            weight = np.random.choice([1, 2], p=[1 - (weight - int(weight)), weight - int(weight)])

            with open(data_file, 'r') as f:
                for line in f:
                    try:
                        example = json.loads(line)

                        # check if id is already in temp_ids
                        if example['id'] in temp_ids:
                            continue

                        temp_ids.add(example['id'])

                        self.data.append(example)
                        for _ in range(weight):
                            self.index.append(len(self.data) - 1)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):

        example = self.data[self.index[idx]]
        image = load_image(example['img_url'])

        if image is None:
            return self[(idx+1) % len(self)]

        # print(image.shape)
        question = f"title: \"{example['title']}\"."
        answer = graph2text(example['comments'])
        
        # Use the BLIP processor to prepare the inputs
        inputs = self.processor(image, question, return_tensors="pt")
        
        # squeeze 0th dimension from all tensors
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # print(inputs['pixel_values'].shape, inputs['input_ids'].shape, inputs['attention_mask'].shape)

        # tokenizer answer
        # answer = self.processor.tokenizer(answer, return_tensors="pt")['input_ids'].squeeze(0)
        answer = processor.tokenizer(answer, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")['input_ids'].squeeze(0)

        
        return inputs, answer

from torch.nn.utils.rnn import pad_sequence
import torch
def vqa_collate_fn(batch):
    # Unzip the batch data
    inputs_list, answers_list = zip(*batch)
    
    # Handle the image pixel values and textual inputs
    pixel_values = torch.stack([item['pixel_values'] for item in inputs_list])
    
    # Prepare input_ids and ensure attention_mask aligns with it
    input_ids = pad_sequence([item['input_ids'] for item in inputs_list], batch_first=True, padding_value=0)
    attention_mask = torch.zeros_like(input_ids)  # Create a zero mask
    attention_mask[input_ids != 0] = 1  # Set mask to 1 where input_ids are not zero
    
    # Stack the answers and pad them
    answers = pad_sequence(answers_list, batch_first=True, padding_value=0)
    
    # Combine the inputs into a single dictionary
    inputs = {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    return inputs, answers


# Example usage
data_files = [
    {'file': '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_ethical.jsonl', 'weight': 1.5},
    {'file': '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_intlpolitics.jsonl', 'weight': 1.2},
    {'file': '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_legal.jsonl', 'weight': 1.2},
    {'file': '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_social.jsonl', 'weight': 1.2},
    {'file': '/home/andre/discursive/caption/data/reddit_discourses/reddit_3x3discourse_affective.jsonl', 'weight': 1},
]
processor = BlipProcessor.from_pretrained(FROM_PRETRAINED, max_position_embeddings = MAX_SEQ_LEN)
dataset = VQADataset(data_files, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=vqa_collate_fn)



import torch
from transformers import BlipForQuestionAnswering, AdamW, BlipConfig, BlipTextConfig

# Load the model
model = BlipForQuestionAnswering.from_pretrained(LOAD_WEIGHTS).to(ACCELERATOR)



'''
TRAINING
'''

model.train()

# Define optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Define loss function, assuming labels are encoded as IDs
criterion = torch.nn.CrossEntropyLoss()

from tqdm import tqdm

history = []

# Training loop
for epoch in range(NUM_EPOCHS):

    total_loss = 0
    batches = 0

    print() # Add a newline for better readability

    for inputs, answers in tqdm(dataloader):
        batches += 1

        # Move batch to the same device as model
        inputs = {k: v.to(ACCELERATOR) for k, v in inputs.items()}

        # print({k: v.shape for k, v in inputs.items()})
        # print(answers.shape)

        # print decoded answers for sanity check
        # print(processor.tokenizer.decode(answers[0]))=

        
        # Generate model outputs
        outputs = model(**inputs, labels=answers)
        
        # Compute loss
        loss = outputs.loss
        
        history.append(loss.item())

        # Backpropagate error
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # running loss -- previous 10 batches
        running_loss = sum(history[-10:]) / min(10, len(history))

        print(f"Epoch {epoch+1} | Current loss: {running_loss}")

        # print out sample behavior every 10 batches using generate
        if batches % 10 == 0:

            model.eval()
            generated = model.generate(**inputs, max_length=MAX_SEQ_LEN)
            print("Generated sample #1:", processor.tokenizer.decode(generated[0]))
            print("Generated sample #2:", processor.tokenizer.decode(generated[1]))
            print("Generated sample #3:", processor.tokenizer.decode(generated[2]))

            model.train()
# 
        # if batches % 500 == 0:
            # save to folder called model_checkpoints
    model.save_pretrained(f"checkpoints/model_checkpoints_{EDITION}/epoch-{epoch}")
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    # json dump history to file
    with open("loss_history_{EDITION}.json", "w") as f:
        json.dump(history, f)

