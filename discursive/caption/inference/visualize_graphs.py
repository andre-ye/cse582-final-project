'''
Visualize graph predictions from the VQA model.
'''

PREDICTIONS = '/home/andre/discursive/caption/data/gen_samples/raw/SMID_v4-e13-b1500.jsonl'
OUT_FOLDER = '/home/andre/discursive/caption/data/gen_samples/visual/SMID_v4-e13-b1500'

import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# clear and create out folder
if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)
else:
    for file in os.listdir(OUT_FOLDER):
        os.remove(os.path.join(OUT_FOLDER, file))

# read predictions
preds = pd.read_json(PREDICTIONS, lines=True)



print(preds)

# visualize a generic graph structure
import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

import json
import re

def parse_input(input_str):
    # Convert to a proper JSON-like format
    input_str = "[" + input_str + "]"
    
    # Replace unquoted `none` with `null`
    input_str = re.sub(r'\bnone\b', 'null', input_str)
    
    # Add double quotes around keys by finding the keys at the start of key-value pairs
    input_str = re.sub(r'(?<=\{|\s)(id|reply to|comment)(?=\s*:\s*)', r'"\1"', input_str)
    
    # Properly escape quotes inside the comment strings
    # This regex finds text inside the comment strings and escapes the quotes within them
    input_str = re.sub(r'("comment"\s*:\s*")(.*?)(")(\s*[,}])', lambda m: m.group(1) + m.group(2).replace('"', '\\"') + m.group(3) + m.group(4), input_str)
    
    # replace "reply to" with "reply_to"
    input_str = re.sub(r'"reply to"', r'"reply_to"', input_str)

    print(input_str)  # For debugging purposes
    return json.loads(input_str)

# Function to create the graph from the parsed data
def create_graph(data):
    G = nx.DiGraph()
    for entry in data:
        G.add_node(entry["id"], comment=entry["comment"])
        if entry["reply_to"] is not None:
            G.add_edge(entry["reply_to"], entry["id"])
    return G

# Function to draw text inside square nodes
def draw_text_inside_square(ax, pos, text):
    for node, (x, y) in pos.items():
        comment = text[node]
        bbox_props = dict(boxstyle="round,pad=0.3,rounding_size=0.2", edgecolor="black", linewidth=1, facecolor="lightblue")
        ax.text(x, y, comment, ha='center', va='center', fontsize=8, bbox=bbox_props)



'''
RUN AREA
'''


# Example input string
input_str = """{ id : 0, reply to : none, comment : "if she was never full brrrs, her food looks better." }, { id : 1, reply to : 0, comment : "a miss me generation." }, { id : 2, reply to : 0, comment : "it was the little known fact that she in the middle east is vietnamese. edit : nobody realized this" }, { id : 3, reply to : 2, comment : "two years" }, { id : 4, reply to : 0, comment : "probably a little kid." }, { id : 5, reply to : 4, comment : "pretty sure she is in west point during this month." }, { id : 6, reply to : none, comment : "i mean no one is reading on the menu... oh jesus, okay she's probably good at passing." }, { id : 7, reply to : 6, comment : "this. it's because she hates meat." }, { id : 8, reply to : 7, comment : "what would you like the little baby bits??" }, { id : 9, reply to : 6, comment : "so romantic... or i'll have to read to the manager of ma's restaurant" }, { id : 10, reply to : 6, comment : "i've had to put a little debbie downer about who she actually ate on the menu. i've never eaten there, but now i look up wilford so much. it's too much better than letting a 2 - year - old girl eat a meal that she's had on her menu." }, { id : 11, reply to : 10, comment : "you remind me of john ramen for my chicken fingers. you know, those aren't chicken wings'more than half way." }, { id : 12, reply to : 10, comment : "how do you know it's not a little * scratch *?" }, { id : 13, reply to : none, comment : "it looks like someone else had to eat it and nobody cares but think it was a good look at least a person. a few people are eating it because i know why so much they're negative." }, { id : 14, reply to : 13, comment : "i mean it\'s way"}"""


data = parse_input(input_str)
G = create_graph(data)


# Get positions for the nodes
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Get node attributes
node_text = nx.get_node_attributes(G, 'comment')

# Plot the graph
plt.figure(figsize=(14, 14))
ax = plt.gca()
nx.draw(G, pos, with_labels=False, node_size=0, node_color="none", edge_color="gray", ax=ax, arrowsize=20)

# Draw text inside fancy boxes
for node, (x, y) in pos.items():
    comment = node_text[node]
    width = max(3, len(comment) * 0.02)  # Width based on text length
    height = 1  # Adjusted height for compactness
    fancy_box = FancyBboxPatch((x - width / 2, y - height / 2),
                               width, height,
                               boxstyle="round,pad=0.3,rounding_size=0.2",
                               edgecolor="black", facecolor="lightblue", linewidth=1)
    ax.add_patch(fancy_box)
    ax.text(x, y, comment, ha='center', va='center', fontsize=6, wrap=True)

plt.title("Graph with Text Boxes as Nodes")
plt.axis('off')  # Turn off the axis
plt.savefig('test.png')  # Save the figure as an image file