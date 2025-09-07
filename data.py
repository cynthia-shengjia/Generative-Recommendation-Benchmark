import json
import random
from torch.utils.data import Dataset
import os
class SeqRecDataset(Dataset):
    """
    专门为序列推荐任务设计的数据集类。
    - 它处理用户的历史行为序列。
    - 它将每个 item_id 转换为4个token的表示。
    """
    def __init__(self, args, mode="train"):
        self.args = args
        self.mode = mode
        
        # 1. 加载 item 到 4-token 的映射文件
        item2tokens_file = os.path.join(args.data_path, args.dataset, "newitem2tokens.json")
        print(f"Loading item-to-4-tokens mapping from {item2tokens_file}...")
        with open(item2tokens_file, 'r') as f:
            self.item2tokens = json.load(f)

        # 2. 加载用户序列数据
        sequence_file = os.path.join(args.data_path, args.dataset, "interaction.json")
        print(f"Loading ALL user sequences from a single source: {sequence_file}...")
        self.user_sequences = self._load_sequences(sequence_file)

        # 3. 将序列数据转换为训练/验证/测试样本
        self.samples,sample_num = self._create_samples()
        if sample_num < 0 or sample_num == 0:
            print(f"sample_num:",sample_num)
        if self.mode != 'test':
            self.samples = random.sample(self.samples, sample_num)
        print(f"Created {len(self.samples)} samples for {self.mode} mode.")

    def _load_sequences(self, file_path):
        """
        从 JSON 文件加载序列数据。
        JSON 格式: {"user_id": [item_id_1, item_id_2, ...]}
        """
        sequences = {}
        with open(file_path, 'r') as f:
            raw_sequences = json.load(f)
        for user_id, item_ids in raw_sequences.items():
            processed_items = [
                str(item) for item in item_ids if str(item) in self.item2tokens
            ]
            sequences[user_id] = processed_items
            
        return sequences

    def _create_samples(self):
        samples = []
        samples_num = 0
        for user_id, item_sequence in self.user_sequences.items():
            if len(item_sequence) < 3:
                continue

            if self.mode == 'train':
                train_seq = item_sequence[:-2]
                for i in range(1, len(train_seq)):
                    target = train_seq[i]
                    history = train_seq[:i]
                    if self.args.max_his_len > 0:
                        history = history[-self.args.max_his_len:]
                    if len(history) > 0:
                        samples.append({"history_items": history, "target_item": target})
                        samples_num +=1
                # history = item_sequence[:-3]
                # if self.args.max_his_len > 0:
                #     history = history[-self.args.max_his_len:]
                # target = item_sequence[-3]
                # samples.append({"history_items": history, "target_item": target})

            elif self.mode == 'valid':
                history = item_sequence[:-2]
                if self.args.max_his_len > 0:
                    history = history[-self.args.max_his_len:]
                target = item_sequence[-2]
                if len(history) > 0: # 确保历史不为空
                    samples.append({"history_items": history, "target_item": target})
                    samples_num +=1
            elif self.mode == 'test':
                history = item_sequence[:-1]
                if self.args.max_his_len > 0:
                    history = history[-self.args.max_his_len:]
                target = item_sequence[-1]
                if len(history) > 0: # 确保历史不为空
                    samples.append({"history_items": history, "target_item": target})
                    samples_num +=1
        return samples,samples_num

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        history_items = sample["history_items"]
        target_item = sample["target_item"]

        # 将 history item_ids 转换为一个扁平化的token列表
        source_tokens = []
        for item_id in history_items:
            source_tokens.extend(self.item2tokens.get(item_id, [0,0,0,0]))

        # 将 target item_id 转换为4个token
        target_tokens = self.item2tokens.get(target_item, [0,0,0,0])

        return {
            "source_tokens": source_tokens,
            "target_tokens": target_tokens
        }