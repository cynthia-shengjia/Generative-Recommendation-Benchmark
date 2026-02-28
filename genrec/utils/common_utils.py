import torch
import numpy as np
import random
import ast
import json 


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def tokens_to_item_id(
    tokens_sequence, 
    tokens_to_item_map
):

    if torch.is_tensor(tokens_sequence):
        tokens_list = tokens_sequence.tolist()
    else:
        tokens_list = tokens_sequence
    

    tokens_tuple = tuple(tokens_list)
    

    return tokens_to_item_map.get(tokens_tuple, None)



def create_token_to_item_mapping(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    token_to_item_map = {}
    
    for key, value in data.items():
        try:
            tokens_tuple = ast.literal_eval(key)
            tokens_list = list(tokens_tuple)
        except:
            cleaned_key = key.strip('()').replace(' ', '')
            tokens_list = [int(x) for x in cleaned_key.split(',') if x]
        

        token_to_item_map[tuple(tokens_list)] = value

    return token_to_item_map

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


