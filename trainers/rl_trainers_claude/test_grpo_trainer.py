# test_grpo_trainer.py (完整版本)

import torch
from torch.utils.data import Dataset
from transformers import T5Config, T5ForConditionalGeneration
from trl import GRPOConfig
from typing import Dict, List
import random

def create_t5_model(vocab_size: int, model_config: dict) -> T5ForConditionalGeneration:
    """
    创建标准的T5模型，根据提供的配置参数
    """
    config = T5Config(
        vocab_size=vocab_size,
        d_model=model_config['d_model'],
        d_kv=model_config['d_kv'],
        d_ff=model_config['d_ff'],
        num_layers=model_config['num_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        num_heads=model_config['num_heads'],
        dropout_rate=model_config['dropout_rate'],
        tie_word_embeddings=model_config['tie_word_embeddings'],
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    
    model = T5ForConditionalGeneration(config)
    return model

class DummyGenRecDataset(Dataset):
    """Dummy dataset for testing"""
    
    def __init__(self, num_samples: int, max_seq_len: int, item2tokens: Dict[int, List[int]]):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.item2tokens = item2tokens
        self.item_ids = list(item2tokens.keys())
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create random history sequence
        history_length = random.randint(3, 10)
        history_items = random.choices(self.item_ids, k=history_length)
        
        # Flatten history tokens
        input_tokens = []
        for item_id in history_items:
            input_tokens.extend(self.item2tokens[item_id])
        
        # Pad or truncate
        if len(input_tokens) > self.max_seq_len:
            input_tokens = input_tokens[-self.max_seq_len:]
        else:
            input_tokens = [0] * (self.max_seq_len - len(input_tokens)) + input_tokens
        
        # Create attention mask
        attention_mask = [1 if t != 0 else 0 for t in input_tokens]
        
        # Random target item
        target_item = random.choice(self.item_ids)
        target_tokens = self.item2tokens[target_item]
        
        # Labels: just the item tokens (no BOS, EOS will be added during generation)
        labels = target_tokens + [1]  # Add EOS
        
        return {
            "input_ids": torch.tensor(input_tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def test_grpo_trainer():
    """Test the GRPO trainer with a small T5 model"""
    
    print("=" * 80)
    print("Testing GRPO Trainer for Generative Recommendation")
    print("=" * 80)
    
    # 1. Create item2tokens mapping (3-level codebook)
    print("\n1. Creating item2tokens mapping...")
    item2tokens = {
        1: [1, 178, 234],
        2: [2, 169, 278],
        3: [3, 188, 299],
        4: [4, 192, 276],
        5: [5, 201, 323],
        6: [6, 150, 250],
        7: [7, 160, 260],
        8: [8, 170, 270],
        9: [9, 180, 280],
        10: [10, 190, 290],
    }
    print(f"Created mapping for {len(item2tokens)} items")
    print(f"Example: Item 1 -> {item2tokens[1]}")
    
    # 2. Build Trie
    print("\n2. Building Trie for constrained generation...")
    from GRPOTrainer import Trie, prefix_allowed_tokens_fn
    
    candidate_trie = Trie(item2tokens)
    print(f"Trie built with {len(candidate_trie)} sequences")
    
    # Test Trie - 注意：现在不添加BOS前缀
    print("\n2.1 Testing Trie structure...")
    test_prefix_empty = []
    allowed_tokens_empty = candidate_trie.get(test_prefix_empty)
    print(f"Allowed tokens after empty prefix: {allowed_tokens_empty}")
    
    test_prefix_1 = [1]  # 第一个token
    allowed_tokens_1 = candidate_trie.get(test_prefix_1)
    print(f"Allowed tokens after prefix {test_prefix_1}: {allowed_tokens_1}")
    
    test_prefix_2 = [1, 178]  # 前两个token
    allowed_tokens_2 = candidate_trie.get(test_prefix_2)
    print(f"Allowed tokens after prefix {test_prefix_2}: {allowed_tokens_2}")
    
    # Test prefix_allowed_fn
    print("\n2.2 Testing prefix_allowed_tokens_fn...")
    prefix_fn = prefix_allowed_tokens_fn(candidate_trie)
    
 
 
    # 3. Create T5 model
    print("\n3. Creating T5 model...")
    vocab_size = 512  # Covers all codebook ranges (0-511)
    model_config = {
        'd_model': 128,
        'd_kv': 16,
        'd_ff': 256,
        'num_layers': 2,
        'num_decoder_layers': 2,
        'num_heads': 4,
        'dropout_rate': 0.1,
        'tie_word_embeddings': False,
    }
    
    model = create_t5_model(vocab_size, model_config)
    print(f"Model created with vocab_size={vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model config - pad_token_id: {model.config.pad_token_id}")
    print(f"Model config - eos_token_id: {model.config.eos_token_id}")
    print(f"Model config - decoder_start_token_id: {model.config.decoder_start_token_id}")
    
    # 4. Create dataset
    print("\n4. Creating dataset...")
    max_seq_len = 50
    train_dataset = DummyGenRecDataset(
        num_samples=20,  # Small dataset for testing
        max_seq_len=max_seq_len,
        item2tokens=item2tokens
    )
    print(f"Dataset created with {len(train_dataset)} samples")
    
    # Test dataset
    sample = train_dataset[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample labels shape: {sample['labels'].shape}")
    print(f"Sample input_ids (last 10): {sample['input_ids'][-10:].tolist()}")
    print(f"Sample labels: {sample['labels'].tolist()}")
    
    # 5. Create GRPO config
    print("\n5. Creating GRPO configuration...")
    grpo_config = GRPOConfig(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=1,
        save_steps=100,
        max_prompt_length=max_seq_len,
        max_completion_length=4,  # decoder_start + 3 tokens + EOS + padding
        num_generations=2,  # Must divide batch_size
        beta=0.1,  # KL penalty coefficient
        temperature=1.0,
        seed=42,
        report_to=[],  # Disable wandb for testing
    )
    print("GRPO config created")
    
    # 6. Define reward function
    print("\n6. Defining reward function...")
    def custom_reward_func(generated_items: List[int], target_items: List[int]) -> List[float]:
        """Reward 1.0 for correct item, 0.0 otherwise"""
        rewards = []
        for gen_item, target_item in zip(generated_items, target_items):
            if gen_item == target_item:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
    
    print("Reward function defined")
    
    # 7. Create trainer
    print("\n7. Creating GRPO Trainer...")
    try:
        from GRPOTrainer import GRPOTrainerForGenRec
        
        trainer = GRPOTrainerForGenRec(
            model=model,
            item2tokens=item2tokens,
            args=grpo_config,
            train_dataset=train_dataset,
            reward_func=custom_reward_func,
        )
        print("✓ Trainer created successfully!")
        
        # Check trainer attributes
        print(f"  - Trainer has {len(trainer.item2tokens)} items in mapping")
        print(f"  - Trainer has {len(trainer.tokens2item)} token sequences in reverse mapping")
        print(f"  - Trie has {len(trainer.candidate_trie)} sequences")
        
    except Exception as e:
        print(f"✗ Error creating trainer: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 8. Test training step
    print("\n8. Testing training step...")
    try:
        # Get a batch
        from torch.utils.data import DataLoader
        dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        
        # Test _prepare_inputs
        print("\n9. Testing _prepare_inputs...")
        prepared_inputs = trainer._prepare_inputs(batch)
        print(f"✓ Prepared inputs successfully!")
        print(f"  - encoder_input_ids shape: {prepared_inputs['encoder_input_ids'].shape}")
        print(f"  - decoder_input_ids shape: {prepared_inputs['decoder_input_ids'].shape}")
        print(f"  - advantages shape: {prepared_inputs['advantages'].shape}")
        print(f"  - rewards mean: {prepared_inputs['sliced_rewards'].mean().item():.4f}")
        print(f"  - rewards std: {prepared_inputs['sliced_rewards'].std().item():.4f}")
        
        # Test compute_loss
        print("\n10. Testing compute_loss...")
        loss = trainer.compute_loss(model, prepared_inputs)
        print(f"✓ Loss computed successfully!")
        print(f"  - Loss value: {loss.item():.4f}")
        
        # Test metrics
        print("\n11. Checking metrics...")
        for key, values in trainer._metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                print(f"  - {key}: {avg_value:.4f}")
        
        # Test backward pass
        print("\n12. Testing backward pass...")
        loss.backward()
        print(f"✓ Backward pass successful!")
        
        # Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        if has_grad:
            print(f"✓ Gradients computed successfully!")
        else:
            print(f"⚠ Warning: No gradients found")
        
        print("\n" + "=" * 80)
        print("✓ All tests passed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_trie_with_real_data():
    """Test Trie with data format similar to your real use case"""
    print("\n" + "=" * 80)
    print("Testing Trie with Real Data Format")
    print("=" * 80)
    
    # Use format similar to your real data
    item2tokens = {
        1: [171, 357, 778, 869],
        2: [262, 579, 619, 868],
        3: [120, 569, 801, 868],
        4: [182, 608, 706, 868],
        5: [171, 457, 807, 868],
    }
    
    print(f"\n1. Item2tokens mapping:")
    for item_id, tokens in item2tokens.items():
        print(f"  Item {item_id}: {tokens}")
    
    # Build Trie
    from GRPOTrainer import Trie, prefix_allowed_tokens_fn
    
    candidate_trie = Trie(item2tokens)
    prefix_fn = prefix_allowed_tokens_fn(candidate_trie)
    
    # Test sequences
    test_sequences = [
        [0],
        [0, 171],
        [0, 171, 357, 778]
    ]
    
    for seq in test_sequences:
        allowed = prefix_fn(0, torch.tensor(seq))
        print(f"  {seq} -> {sorted(allowed) if allowed else 'empty'}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Test Trie with real data format first
    # test_trie_with_real_data()
    
    # Run main tests
    # print("\n")
    test_grpo_trainer()