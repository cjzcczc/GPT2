import torch
from GPT2 import GPTconfig,GPT
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from data import DataLoader


ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    master_process = rank == 0
else:
    rank = 0
    local_rank = 0
    world_size = 1
    master_process = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_lr = 2.5e-4
min_lr = max_lr / 10   
warmup_steps = 100
max_steps = 1000
def getlr(step:int):
    '''
    step 当前训练步数 
    从1开始
    '''
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    elif step > max_steps:
        return min_lr   
    decay_ratio =  (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + torch.cos(torch.tensor(decay_ratio) * torch.pi))
    return min_lr + (max_lr - min_lr) * coeff

total_batch_size = 524288
B = 16
T = 1024

grad_accum_steps = total_batch_size // (B*T*world_size)
if master_process:
    print(f"total_batch_size: {total_batch_size}")
    print(f"=> grad_accum_steps: {grad_accum_steps}")

model = GPT(GPTconfig())
model = model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[local_rank])
raw_model = model.module if ddp else model
DataLoaderLite = DataLoader(
    data_path = "data.txt",
    batch_size = B,
    seq_len = T,
    process_rank = rank,
    num_processes = world_size
)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learing_rate=max_lr, device=device)
for step in range(max_steps):
    optimizer.zero_grad()
    loss_accum = 0.0
    for mircostep in range(grad_accum_steps):
        x, y = DataLoaderLite.get_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits,loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (mircostep == grad_accum_steps - 1)       
            loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = getlr(step+1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if master_process and step % 10 == 0:
        print(f"step{step}, loss: {loss_accum}, lr: {lr}, grad_norm: {norm}")
if ddp:
    destroy_process_group()