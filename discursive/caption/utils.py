import numpy as np
import requests
from PIL import Image
from io import BytesIO

def load_image(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        # Send an HTTP GET request to the URL with headers
        response = requests.get(url, headers=headers)
        # Check if the request was successful
        response.raise_for_status()

        # Open the image from the bytes of the response
        image = Image.open(BytesIO(response.content))

        # Convert the image to RGB format to ensure compatibility
        image = image.convert('RGB')

        # Convert the image to a NumPy array
        numpy_array = np.array(image)

        return numpy_array
    except:
        return None

def graph2text(graph):
    running_text = ""
    
    queue = []
    
    def recurse_queue_add(comment_tree, parent_id):
        queue.append({
            'parent_id': parent_id,
            'id': comment_tree['id'],
            'author': comment_tree['author'],
            'body': comment_tree['body']
        })
        for reply in comment_tree['replies']:
            recurse_queue_add(reply, comment_tree['id'])
            
    for main_level_comment in graph:
        recurse_queue_add(main_level_comment, 'None')
            
    authors = dict.fromkeys([qitem['author'] for qitem in queue])
    authors_map = {j:i for i, j in enumerate(authors)}
    post_ids = dict.fromkeys([qitem['id'] for qitem in queue] + [qitem['parent_id'] for qitem in queue])
    post_ids_map = {j:i for i, j in enumerate(post_ids)}
    post_ids_map['None'] = 'None'
    
    for i in range(len(queue)):
        queue[i]['parent_id'] = post_ids_map[queue[i]['parent_id']]
        queue[i]['id'] = post_ids_map[queue[i]['id']]
        queue[i]['author'] = authors_map[queue[i]['author']]
        
    for item in queue:
        item_str = "{"
        item_str += f"id: {item['id']}, "
        item_str += f"reply to: {item['parent_id']}, "
        item_str += f"comment: \"{item['body']}\""
        item_str += "}"
        running_text += item_str + ", "
        
    return running_text