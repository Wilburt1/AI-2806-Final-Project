import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTLanguageModel
import argparse
import string
from utils import get_batch, estimate_loss, levenshtein_distance, get_stats, merge, decode_token_basic, encode_text_token_basic
import tiktoken
import uuid  # Added to generate UUID for artifact_id

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--target_vocab_size', type=int, default=400,
                        help='target vocabulary size')
    parser.add_argument('--block_size', type=int, default=256,
                        help='block size for processing vocabulary')
    
    parser.add_argument('--max_iters', type=int, default=1000,
                        help='maximum number of training iterations')
    parser.add_argument('--eval_iters', type=int, default=30,
                        help='number of evaluation iterations')

    parser.add_argument('--eval_interval', type=int, default=256,
                        help='evaluation interval')
    
    # Optimization parameters
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay')
    
    # Model and dataset
    parser.add_argument('--model', type=str, default='basic')
    parser.add_argument('--tokenization_strategy', type=str, default='')
    parser.add_argument('--save_file', type=str, default='./models/final_model.pth')
    # Modified: Specify the pretrained model path
    parser.add_argument('--ckpt', type=str, default='./models/pretrain.pth',
                       help='pretrained model checkpoint path')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--n_heads', type=int, default=6, help='number of attention heads')
    parser.add_argument('--n_layer', type=int, default=6, help='number of attention layers')
    parser.add_argument('--n_embd', type=int, default=384, help='embedding dimension')
    parser.add_argument('--loss', type=str, default='NLL')
    parser.add_argument('--training_file', type=str, default='./train_data/Dataset_improved.txt')
    parser.add_argument('--training_file_tokenizer', type=str, default='./train_data/Dataset_improved.txt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Dataset_improved', choices=['Dataset_improved'], help='dataset')
    
    # Fine-tuning parameters
    parser.add_argument('--fine_tune', action='store_true', help='whether to perform fine-tuning')
    parser.add_argument('--fine_tune_file', type=str, default='./train_data/QA.txt')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5, help='fine-tuning learning rate')
    parser.add_argument('--fine_tune_weight_decay', type=float, default=0.1, help='fine-tuning weight decay')

    opt = parser.parse_args()
    return opt

def train_model(opt, data, vocab_size, phase='finetune'):
    # Split data into train and validation sets
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Initialize model
    model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
    
    # Modified: Load the pretrained model from the specified checkpoint
    checkpoint_path = opt.ckpt
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded pretrained model from {checkpoint_path}")
    
    model = model.to(opt.device)

    # Modified: Use fine-tuning specific learning rate and weight decay
    lr = opt.fine_tune_lr
    wd = opt.fine_tune_weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # Training loop
    for iter in range(opt.max_iters):
        if iter % opt.eval_interval == 0:
            losses = estimate_loss(opt, model, train_data, val_data)
            print(f"Phase: {phase}, Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data, val_data, opt)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the fine-tuned model
    save_path = opt.save_file
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved to {save_path}")
    return model

def main():
    opt = parse_option()

    # Modified: Directly load the fine-tuning dataset
    with open(opt.fine_tune_file, 'r', encoding='utf-8') as f:
        fine_tune_text = f.read()

    # Modified: Load tokenizer data to ensure consistent tokenization
    with open(opt.training_file_tokenizer, 'r', encoding='utf-8') as f:
        text_token = f.read()

    # Tokenization
    if opt.tokenization_strategy == '':
        ascii = string.printable
        chars = list(ascii)
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        fine_tune_data = torch.tensor(encode(fine_tune_text), dtype=torch.long)
    elif opt.tokenization_strategy == 'BPE':
        tokens = text_token.encode('utf-8')
        desired_vocab_size = opt.target_vocab_size
        num_merges = desired_vocab_size - 256
        ids = list(tokens)
        merges = {}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        vocab_size = desired_vocab_size
        fine_tune_data = torch.tensor(encode_text_token_basic(fine_tune_text, merges), dtype=torch.long)
    elif opt.tokenization_strategy == 'GPT2':
        enc = tiktoken.get_encoding('gpt2')
        vocab_size = 50257
        fine_tune_data = torch.tensor(enc.encode(fine_tune_text), dtype=torch.long)

    # Modified: Skip pretraining and directly perform fine-tuning
    print("Starting fine-tuning phase...")
    model = train_model(opt, fine_tune_data, vocab_size, phase='finetune')

if __name__ == "__main__":
    main()