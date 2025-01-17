import os
import pickle

import numpy as np
import torch

from model import GPT, GPTConfig

device = 'cpu'
model_compile = True
dataset = 'tinyshakespeare'
data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
meta_vocab_size = meta['vocab_size']
checkpoint_dir = 'checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)

batch_size = 12
eval_iters = 50

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

# model 参数, init
block_size = 64
model_args = dict(
    n_layer=4,
    n_head=4,
    n_embd=128,
    block_size=block_size,
    bias=False,
    vocab_size=meta_vocab_size,
    dropout=0.0
)

# 有 checkpoint 则从 checkpoint 恢复
ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    checkpoint_model_args = checkpoint['model_args']
    # 这些参数是不能改的, 强制指定要恢复
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    # train checkpoint 参数
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    gptconf = GPTConfig(**model_args)
    # train checkpoint 参数
    iter_num = 0
    best_val_loss = 2.0
    model = GPT(gptconf)

# compile the model
if model_compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

torch.set_num_threads(24)
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
while True:
    X, Y = get_batch('train')
    logits, loss = model(X, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # losses = estimate_loss()
    if iter_num % 100 == 0:
        losses = estimate_loss()
        print(f"\rstep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", end='', flush=True)
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config
            }
            print(f"\nsaving checkpoint to {checkpoint_dir}", flush=True)
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'ckpt.pt'))
    iter_num += 1
    if iter_num > 600000:
        break
