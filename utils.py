import json
import torch
# 读取JSON文件
def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# 读取JSON文件并创建映射字典
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
        
        # 创建映射
        token_to_item_map[tuple(tokens_list)] = value

    return token_to_item_map

# 将tokens转化为item_ID
def tokens_to_item_id(tokens_sequence, tokens_to_item_map):
    # 如果是张量，转换为列表
    if torch.is_tensor(tokens_sequence):
        tokens_list = tokens_sequence.tolist()
    else:
        tokens_list = tokens_sequence
    
    # 转换为tuple作为字典的key
    tokens_tuple = tuple(tokens_list)
    
    # 查找对应的item ID
    return tokens_to_item_map.get(tokens_tuple, None)



def calc_ndcg(rank, k):
    """
    计算NDCG值
    
    参数:
        rank: 真实物品在推荐列表中的位置(从1开始)
        k: 推荐列表长度
    
    返回:
        NDCG值
    """
    if rank <= k:
        return 1 / math.log2(rank + 1)
    return 0