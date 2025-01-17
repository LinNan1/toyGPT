import os
import pickle

import torch

from model import GPTConfig, GPT

device = 'cpu'
model_args = dict()
checkpoint_dir = 'checkpoint'
# 有 checkpoint 则从 checkpoint 恢复
ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    # load 词库
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    # load 模型
    checkpoint_model_args = checkpoint['model_args']
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

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

start = "\n"
start_ids = encode(start)

num_samples = 1
max_new_tokens = 500
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
with torch.no_grad():
    for k in range(num_samples):
        y = model.generate(x, max_new_tokens)
        print(decode(y[0].tolist()))
        print('---------------')