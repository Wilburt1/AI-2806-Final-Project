import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTLanguageModel
import argparse
import string
from utils import get_batch, estimate_loss, levenshtein_distance, get_stats, merge, decode_token_basic, encode_text_token_basic
import tiktoken

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--target_vocab_size', type=int, default=400,
                        help='目标词汇表大小')
    parser.add_argument('--block_size', type=int, default=256,
                        help='处理词汇的块大小')
    
    parser.add_argument('--max_iters', type=int, default=1000,
                        help='训练过程最大迭代次数')
    parser.add_argument('--eval_iters', type=int, default=30,
                        help='评估迭代次数')

    parser.add_argument('--eval_interval', type=int, default=256,
                        help='评估间隔')
    
    # 优化参数
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='丢弃率')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='动量')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='权重衰减')
    
    # 模型和数据集
    parser.add_argument('--model', type=str, default='basic')
    parser.add_argument('--tokenization_strategy', type=str, default='')
    # 更新默认保存路径到指定模型目录
    parser.add_argument('--save_file', type=str, default='./models/final_model.pth')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--n_heads', type=int, default=6, help='注意力头数')
    parser.add_argument('--n_layer', type=int, default=6, help='注意力层数')
    parser.add_argument('--n_embd', type=int, default=384, help='嵌入维度')
    parser.add_argument('--loss', type=str, default='NLL')
    # 更新默认数据集路径
    parser.add_argument('--training_file', type=str, default='./train_data/Dataset_improved.txt')
    parser.add_argument('--training_file_tokenizer', type=str, default='./train_data/Dataset_improved.txt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Dataset_improved',choices=['Dataset_improved'], help='dataset')
    
    # 微调参数
    parser.add_argument('--fine_tune', action='store_true', help='是否进行微调')
    # 更新微调数据集路径
    parser.add_argument('--fine_tune_file', type=str, default='./train_data/QA.txt')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5, help='微调学习率')
    parser.add_argument('--fine_tune_weight_decay', type=float, default=0.1, help='微调权重衰减')

    opt = parser.parse_args()
    return opt

def train_model(opt, data, vocab_size, phase='pretrain'):
   
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
    
    # 如果是微调阶段或指定了检查点，加载预训练模型
    if phase == 'finetune' or opt.ckpt:
        # 更新预训练模型默认路径
        checkpoint_path = opt.ckpt if opt.ckpt else './models/pretrained.pth'
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    model = model.to(opt.device)

    # 根据训练阶段设置学习率和权重衰减
    lr = opt.fine_tune_lr if phase == 'finetune' else opt.learning_rate
    wd = opt.fine_tune_weight_decay if phase == 'finetune' else opt.weight_decay
    
    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    for iter in range(opt.max_iters):
        if iter % opt.eval_interval == 0:
            losses = estimate_loss(opt, model, train_data, val_data)
            print(f"Phase: {phase}, Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data, val_data, opt)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 保存模型
    save_path = './models/pretrained.pth' if phase == 'pretrain' else opt.save_file
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

def main():
    opt = parse_option()

    # 加载预训练数据集
    with open(opt.training_file, 'r', encoding='utf-8') as f:
        text = f.read()

    with open(opt.training_file_tokenizer, 'r', encoding='utf-8') as f:
        text_token = f.read()

    # 分词器处理
    if opt.tokenization_strategy == '':
        ascii = string.printable
        chars = list(ascii)
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        data = torch.tensor(encode(text), dtype=torch.long)
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
        data = torch.tensor(encode_text_token_basic(text_token, merges), dtype=torch.long)
    elif opt.tokenization_strategy == 'GPT2':
        enc = tiktoken.get_encoding('gpt2')
        vocab_size = 50257
        data = torch.tensor(enc.encode(text), dtype=torch.long)

    # 第一阶段：预训练
    print("Starting pre-training phase...")
    model = train_model(opt, data, vocab_size, phase='pretrain')

    # 第二阶段：微调
    if opt.fine_tune:
        print("Starting fine-tuning phase...")
        with open(opt.fine_tune_file, 'r', encoding='utf-8') as f:
            fine_tune_text = f.read()
        
        # 使用相同的分词器处理微调数据
        if opt.tokenization_strategy == '':
            fine_tune_data = torch.tensor(encode(fine_tune_text), dtype=torch.long)
        elif opt.tokenization_strategy == 'BPE':
            fine_tune_data = torch.tensor(encode_text_token_basic(fine_tune_text, merges), dtype=torch.long)
        elif opt.tokenization_strategy == 'GPT2':
            fine_tune_data = torch.tensor(enc.encode(fine_tune_text), dtype=torch.long)

        # 微调模型
        train_model(opt, fine_tune_data, vocab_size, phase='finetune')

if __name__ == "__main__":
    main()