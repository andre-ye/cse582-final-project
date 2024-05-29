'''
Reads images in the directory /home/andre/discursive/data/SMID/imgs.
Uses the BLIP model to produce captions of the SMID imgs.
Run on cuda:0 if available, otherwise CPU.

Example usage of BLIP from HuggingFace:

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))


Make sure to check that images end with .jpg in SMD_imgs.
Save captions to a jsonl file in SMID/captions.jsonl
'''

import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# Constants
ACCELERATOR = 'cuda:0' if torch.cuda.is_available() else 'cpu'
output_file = '/home/andre/discursive/data/SMID/captions.jsonl'

# Load the BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(ACCELERATOR)

# Load the images
image_dir = '/home/andre/discursive/data/SMID/imgs'
image_files = os.listdir(image_dir)
image_files = [f for f in image_files if f.endswith('.jpg')]

# Generate captions
captions = {'img_name': [], 'caption': []}
for image_file in tqdm(image_files):
    image_file_path = os.path.join(image_dir, image_file)
    raw_image = Image.open(image_file_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(ACCELERATOR)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    captions['img_name'].append(image_file)
    captions['caption'].append(caption)

# Save captions to a jsonl file
with open(output_file, 'w') as file:
    for img_name, caption in zip(captions['img_name'], captions['caption']):
        file.write(json.dumps({'img_name': img_name, 'caption': caption}) + '\n')