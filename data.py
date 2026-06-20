import os

data_path = os.path.join(os.path.dirname(__file__), 'data', 'input.txt')

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

import torch

def get_batch(split):
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    data = data[:n] if split == 'train' else data[n:]
    ix = torch.randint(len(data) - 1 - 256, (64,))
    x = torch.stack([data[i:i+256] for i in ix])
    y = torch.stack([data[i+1:i+1+256] for i in ix])
    return x, y
