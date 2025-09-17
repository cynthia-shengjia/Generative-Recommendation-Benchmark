from typing import Dict, List, Tuple, Optional

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self, item2tokens: Optional[Dict[int, Tuple]] = None):
        self.root = TrieNode()
        self.len = 0
        
        if item2tokens:
            sequences = self.add_prefix(item2tokens)
            for sequence in sequences:
                self.insert(sequence)
                self.len += 1

    def add_prefix(self, item2tokens: Dict[int, Tuple]):
        """添加前缀到物品token序列"""
        prefix_added_items = [[0] + list(items) for _, items in item2tokens.items()]
        return prefix_added_items

    def insert(self, token_list: List[int]):
        """插入token序列到Trie"""
        node = self.root
        for token in token_list:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end_of_word = True
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        """获取给定前缀后允许的token（用于生成时的前缀约束）"""
        return self.next_tokens(prefix_sequence)

    def next_tokens(self, prefix_list: List[int]):
        """获取给定前缀后允许的token"""
        node = self.root
        for token in prefix_list:
            if token not in node.children:
                return []
            node = node.children[token]
        return list(node.children.keys())

    def valid_tokens(self, token_list: List[int]):
        """获取序列每一步的有效token（用于训练时的约束softmax）"""
        valid_tokens_list = [list(self.root.children.keys())]
        node = self.root
        for token in token_list:
            if token in node.children:
                node = node.children[token]
                valid_tokens_list.append(list(node.children.keys()))
            else:
                valid_tokens_list.append([])
        return valid_tokens_list

    def __iter__(self):
        """迭代所有序列"""
        def _traverse(prefix_sequence, node):
            if node.is_end_of_word:
                yield prefix_sequence
            for token, child_node in node.children.items():
                yield from _traverse(prefix_sequence + [token], child_node)

        return _traverse([], self.root)

    def __len__(self):
        return self.len

def prefix_allowed_tokens_fn(candidate_trie):
    """创建前缀约束函数"""
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens