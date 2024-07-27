import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT
from config import GPT_CONFIG

cfg = GPT_CONFIG()
dataset = 'openwebtext'
data_dir = os.path.join('data', dataset)
torch.manual_seed(1337 + cfg.seed_offset)
torch.backends.cuda.matmul.allow_tf32 = cfg.torch_16bit_allow
torch.backends.cudnn.allow_tf32 = cfg.torch_allow_tf32

print('======================================')
print('       언시 GPT TRAINING PROCESS      ')
print('======================================\n')
print("Initializing a new model from scratch")
print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)\n")
print('---- Config as follow ----\n')
print(cfg)

scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == 'float16'))

model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.embedding_depth, block_size=cfg.block_size,
                  bias=cfg.bias, vocab_size=50257, dropout=cfg.dropout)
gptconf = GPTConfig(**model_args)
print(model_args)
model = GPT(gptconf)
model = model.to(cfg.device)

optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), cfg.device)

if cfg.compile:
    print("compiling the model... (시간이 좀 걸려요..)")
    unoptimized_model = model
    model = torch.compile(model)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]
ctx = nullcontext() if cfg.device == 'cpu' else torch.amp.autocast(device_type=cfg.device, dtype=ptdtype)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join('/media/sien/DATA/CODE/2024/llm/nanoGPT/data/shakespeare', 'train.bin'),
                         dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join('/media/sien/DATA/CODE/2024/llm/nanoGPT/data/shakespeare', 'val.bin'),
                         dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + cfg.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + cfg.block_size]).astype(np.int64)) for i in ix])
    if cfg.device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(cfg.device, non_blocking=True), y.pin_memory().to(cfg.device, non_blocking=True)
    else:
        x, y = x.to(cfg.device), y.to(cfg.device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_interval)
        for k in range(cfg.eval_interval):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


X, Y = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model  # unwrap DDP container if needed
running_mfu = -1.0

# training loop
X, Y = get_batch('train')  # fetch the very first batch
t0 = time.time()
iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
best_val_loss = 1e9
while True:

    # determine and set the learning rate for this iteration
    lr = cfg.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % cfg.eval_interval == 0 :
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss or cfg.save_model:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,

                }
                print(f"saving checkpoint to {cfg.out_dir}")
                torch.save(checkpoint, os.path.join(cfg.out_dir, 'ckpt.pt'))


    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(cfg.gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / cfg.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if cfg.gradient_clipping != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clipping)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % cfg.log_interval == 0 :
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * cfg.gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(cfg.batch_size * cfg.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > cfg.iter:
        break