# PREDICTIONS = '/home/andre/discursive/caption/data/gen_samples/raw/SMID_v4-e13-b1500.jsonl'
PREDICTIONS = '/home/andre/discursive/caption/data/gen_samples/raw/SMID_v7_e51.jsonl'

IMG_DIR = '/home/andre/discursive/data/SMID/imgs'

import pandas as pd
import json
import re

def remove_incomplete_parts(input_str):
    # Match all complete { id : ..., reply to : ..., comment : "..." } patterns
    pattern = re.compile(r'\{ id : \d+, reply to : (none|\d+), comment : ".*?" \}(?:, )?')
    
    # Find all matches and join them back into a single string
    matches = pattern.findall(input_str)
    
    # Construct the result from the matches
    result = []
    for match in matches:
        # Each match is a tuple with "none" or the number, we need to reconstruct the complete string
        id_match = re.search(r'\{ id : \d+, reply to : (none|\d+), comment : ".*?" \}(?:, )?', input_str)
        if id_match:
            result.append(id_match.group())
            input_str = input_str[id_match.end():]  # Move forward in the string

    result = ''.join(result)

    # find the index of the last '}' in the string
    last_brace = result.rfind('}')

    # return the string up to the last brace
    return result[:last_brace+1]
    
    # return ''.join(result)[:-2]

def parse_input(input_str):

    # Remove incomplete parts
    input_str = remove_incomplete_parts(input_str)

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

    # print(input_str)  # For debugging purposes
    return json.loads(input_str)

def build_nested_structure(comments):
    # Create a dictionary to hold the comments by id
    comment_dict = {comment['id']: comment for comment in comments}
    
    # Initialize the root of the nested structure
    nested_structure = {}

    # Iterate through the comments to build the nested structure
    for comment in comments:
        comment['replies'] = []  # Initialize the replies list for each comment
        if comment['reply_to'] is None:
            # Add root-level comments directly to the nested structure
            nested_structure[comment['id']] = comment
        else:
            # Add replies to their respective parent comment
            parent = comment_dict[comment['reply_to']]
            parent['replies'].append(comment)
    
    return nested_structure

def print_nested_structure(nested_structure, indent=0):
    for key, value in nested_structure.items():
        print('\t' * indent + f"{key}: {value['comment']}")
        if value['replies']:
            # Convert list of replies to dict for printing with ids
            replies_dict = {reply['id']: reply for reply in value['replies']}
            print_nested_structure(replies_dict, indent + 1)

# Example usage

import matplotlib.pyplot as plt
import os

def show_pic_from_file(file_path):
    img = plt.imread(file_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

data = pd.read_json(PREDICTIONS, lines=True)

for i in range(len(data)):


    # print image file path
    img_name = data['img_name'][i]
    img_path = os.path.join(IMG_DIR, img_name)
    print(f'image path: {img_path}')
    print()

    input_str = data['graph'][i]
    try:
        parsed_data = parse_input(input_str)
        structured_data = build_nested_structure(parsed_data)
        print_nested_structure(structured_data)
    except:
        print('Error parsing input')
        print(input_str)


    # nested = build_nested_structure(parsed_data)
    # pretty_print(nested)

    # print_nested_structure(nested)

    # print(nested)

    if input('Continue? ') == 'n':
        break

