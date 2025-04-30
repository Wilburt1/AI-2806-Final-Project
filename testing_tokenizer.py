import torch
import torch.nn as nn
import argparse
from model import GPTLanguageModel
from utils import get_batch, estimate_loss, levenshtein_distance, get_stats, merge
from rouge import Rouge
import string
import tiktoken
import pickle
from uuid import uuid4

def parse_option():
    parser = argparse.ArgumentParser('Argument parser for GPT language model training and evaluation')
    
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--block_size', type=int, default=256, help='Context window size for token processing')
    parser.add_argument('--vocab_size', type=int, default=65, help='Vocabulary size for character-based tokenization')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for regularization')
    parser.add_argument('--model', type=str, default='basic', help='Model architecture type')
    parser.add_argument('--tokenization_strategy', type=str, default='', choices=['', 'BPE', 'GPT2'], help='Tokenization strategy')
    parser.add_argument('--ckpt', type=str, default='./models/test.pth', help='Path to model checkpoint')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--loss', type=str, default='NLL', help='Loss function type')
    parser.add_argument('--testing_file_prompt', type=str, default='./test_data/test_prompt_1.txt', help='Path to test prompt file')
    parser.add_argument('--testing_file_response', type=str, default='./test_data/test_response_1.txt', help='Path to test response file')
    parser.add_argument('--testing_file_answer', type=str, default='./test_data/test_answer_1.txt', help='Path to test answer file')
    parser.add_argument('--training_file', type=str, default='./train_data/shakespeare.txt', help='Path to training data')
    parser.add_argument('--generate_token_number', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for computation (cuda:0 or cpu)')
    parser.add_argument('--dataset', type=str, default='shakespeare', choices=['shakespeare'], help='Dataset name')
    
    return parser.parse_args()

def encode_bpe(text, merges, vocab_size=400):
    """Encode text using BPE tokenization."""
    tokens = text.encode('utf-8')
    ids = list(tokens)
    
    # Perform BPE merges
    for pair, idx in merges.items():
        ids = merge(ids, pair, idx)
    
    return ids

def decode_bpe(ids, vocab):
    """Decode BPE token IDs back to text."""
    try:
        tokens = b''.join(vocab[idx] for idx in ids)
        return tokens.decode('utf-8', errors='replace')
    except KeyError as e:
        raise ValueError(f"Invalid token ID in decoding: {e}")

def build_bpe_vocab(text, target_vocab_size=400):
    """Build BPE vocabulary and merges from text."""
    tokens = text.encode('utf-8')
    ids = list(tokens)
    merges = {}
    num_merges = target_vocab_size - 256
    
    # Perform merges
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    
    # Build vocabulary
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    
    return merges, vocab

def main():
    opt = parse_option()
    print(f"Tokenization strategy: '{opt.tokenization_strategy}'")

    # Load training data
    try:
        with open(opt.training_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Training file not found: {opt.training_file}")

    # Tokenization setup
    if opt.tokenization_strategy == '':
        chars = list(string.printable)
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda l: ''.join([itos[i] for i in l if i in itos])
    elif opt.tokenization_strategy == 'BPE':
        try:
            with open('./models/bpe_merges.pkl', 'rb') as f:
                merges = pickle.load(f)
            with open('./models/bpe_vocab.pkl', 'rb') as f:
                vocab = pickle.load(f)
            vocab_size = len(vocab)
            print(f"BPE vocab_size: {vocab_size}")
            encode = lambda s: encode_bpe(s, merges, vocab_size)
            decode = lambda l: decode_bpe(l, vocab)
        except FileNotFoundError:
            print("BPE files missing. Building new BPE vocabulary...")
            merges, vocab = build_bpe_vocab(text, target_vocab_size=400)
            vocab_size = len(vocab)
            with open('./models/bpe_merges.pkl', 'wb') as f:
                pickle.dump(merges, f)
            with open('./models/bpe_vocab.pkl', 'wb') as f:
                pickle.dump(vocab, f)
            encode = lambda s: encode_bpe(s, merges, vocab_size)
            decode = lambda l: decode_bpe(l, vocab)
    elif opt.tokenization_strategy == 'GPT2':
        enc = tiktoken.get_encoding('gpt2')
        vocab_size = 50257
        encode = lambda s: enc.encode(s)
        decode = lambda l: enc.decode(l)
    else:
        raise ValueError(f"Unsupported tokenization strategy: {opt.tokenization_strategy}")

    print(f"Vocabulary size: {vocab_size}")

    # Model setup
    try:
        model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
        model = model.to(opt.device)
        model.load_state_dict(torch.load(opt.ckpt, map_location=opt.device))
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

    # Metrics initialization
    rouge = Rouge()
    levenshtein_distances = []
    rouge_scores = {
        'rouge-1': {'f': [], 'p': [], 'r': []},
        'rouge-2': {'f': [], 'p': [], 'r': []},
        'rouge-l': {'f': [], 'p': [], 'r': []}
    }

    # Process test cases
    for i in range(1, 51):
        prompt_file = f'./test_data/test_prompt_{i}.txt'
        answer_file = f'./test_data/test_answer_{i}.txt'
        response_file = f'./test_data/test_response_{i}.txt'

        try:
            # Load test data
            with open(prompt_file, 'r', encoding='utf-8') as f:
                text_test_prompt = f.read()
            with open(answer_file, 'r', encoding='utf-8') as f:
                text_test_answer = f.read()

            # Generate response
            context = torch.tensor(encode(text_test_prompt), device=opt.device).unsqueeze(0)
            number_gen = len(encode(text_test_answer))
            generated_ids = model.generate(context, max_new_tokens=number_gen, block_size=opt.block_size)[0].tolist()
            
            
            
            response = decode(generated_ids)

            # Save response
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(response)

            # Compute metrics
            lev_dist = levenshtein_distance(text_test_answer, response)
            scores = rouge.get_scores(response, text_test_answer)[0]

            # Store metrics
            levenshtein_distances.append(lev_dist)
            for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                rouge_scores[metric]['f'].append(scores[metric]['f'])
                rouge_scores[metric]['p'].append(scores[metric]['p'])
                rouge_scores[metric]['r'].append(scores[metric]['r'])

            print(f"\nTest Case {i}:")
            print(f"Levenshtein Distance: {lev_dist}")
            print(f"ROUGE Scores: {scores}")

        except FileNotFoundError:
            print(f"Test case {i} skipped: Missing files")
            continue
        except Exception as e:
            print(f"Test case {i} failed: {str(e)}")
            continue

    # Compute and print average metrics
    if levenshtein_distances:
        avg_levenshtein = sum(levenshtein_distances) / len(levenshtein_distances)
        print(f"\nAverage Levenshtein Distance: {avg_levenshtein:.2f}")
    else:
        print("\nNo Levenshtein distances computed.")

    print("\nAverage ROUGE Scores:")
    for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
        if rouge_scores[metric]['f']:
            avg_f = sum(rouge_scores[metric]['f']) / len(rouge_scores[metric]['f'])
            avg_p = sum(rouge_scores[metric]['p']) / len(rouge_scores[metric]['p'])
            avg_r = sum(rouge_scores[metric]['r']) / len(rouge_scores[metric]['r'])
            print(f"{metric}: F1 = {avg_f:.4f}, Precision = {avg_p:.4f}, Recall = {avg_r:.4f}")
        else:
            print(f"{metric}: No scores computed.")

if __name__ == "__main__":
    main()