import os
import pickle

import numpy as np
import torch
import time

from model import GPT, GPTConfig

dataset = 'tinyshakespeare'
data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
meta_vocab_size = meta['vocab_size']

batch_size = 12
block_size = 1024

gptconf = GPTConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=block_size,
    bias=False,
    vocab_size=meta_vocab_size,
    dropout=0.0
)
model = GPT(gptconf)

def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y

X, Y = get_batch('train')
t0 = time.time()
while True:
    lr = 6e-4
    logits, loss = model(X, Y)
