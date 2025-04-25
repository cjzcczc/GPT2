import torch
import math
import torch.nn as nn
import inspect
from dataclasses import dataclass
from torch.nn import functional as F
@dataclass
class GPTconfig:
    block_size : int= 1024
    vocab_size : int= 50257
    n_layer : int= 48
    n_head : int= 25
    n_embd : int= 1600
    attn_pdrop : float= 0.1
    resid_pdrop : float= 0.1
    max_batch_size : int= 32
    use_flash_attn : bool= False

def apply_rope(q, k, seq_len, head_dim, device):
    position_ids = torch.arange(seq_len, dtype=torch.float, device=device)
    div_term = torch.exp(torch.arange(0, head_dim, 2, dtype=torch.float, device=device) * -(math.log(10000.0) / head_dim))
    sinusoid = torch.outer(position_ids, div_term)
    sin, cos = sinusoid.sin(), sinusoid.cos()

    q_rot = torch.cat([q[..., ::2] * cos - q[..., 1::2] * sin, q[..., ::2] * sin + q[..., 1::2] * cos], dim=-1)
    k_rot = torch.cat([k[..., ::2] * cos - k[..., 1::2] * sin, k[..., ::2] * sin + k[..., 1::2] * cos], dim=-1)
    return q_rot, k_rot

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super(CasualSelfAttention, self).__init__()
        self.config = config
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1 / (self.head_dim ** 0.5)
        self.LMAX = 512  # KV Cache 的最大长度
        # KV Cache 初始化为 None，推理时动态分配
        self.use_flash_attn = config.use_flash_attn 
        self.cached_k = None
        self.cached_v = None

    def init_kv_cache(self, batch_size, device):
        """初始化 KV Cache"""
        self.cached_k = torch.zeros(
            batch_size, self.n_head, 0, self.head_dim, device=device
)
        self.cached_v = torch.zeros(
            batch_size, self.n_head, 0, self.head_dim, device=device
)
    
    def forward(self, x, use_cache=False):
        B, T, C = x.size()
        # qkv (B, T, 3*C) -> (B, T, 3, n_head, head_dim)
        # qkv (B, T, 3, n_head, head_dim) -> (3, B, n_head, T, head_dim)
        qkv = self.c_attn(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
                # q,k,v = (B, n_head, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, k, T, self.head_dim, x.device)  # 使用 RoPE 进行位置编码
        if use_cache:
            # 如果缓存为空，初始化缓存
            if self.cached_k is None or self.cached_v is None:
                self.init_kv_cache(B, x.device)

            # 拼接缓存
            self.cached_k = torch.cat([self.cached_k, k], dim=2)  # (B, n_head, T_total, head_dim)
            self.cached_v = torch.cat([self.cached_v, v], dim=2)  # (B, n_head, T_total, head_dim)
            if self.cached_k.size(2) >= self.L_max:
                self.cached_k = self.cached_k[:, :, -self.L_max:, :]  # 保留最后 L_max 个 token 的 K
                self.cached_v = self.cached_v[:, :, -self.L_max:, :]  # 保留最后 L_max 个 token 的 V
            k, v = self.cached_k, self.cached_v

        # 注意力计算
        if self.use_flash_attn:
            y = F.scaled_dot_product_attention(q,k,v,is_causal=True, attn_mask=self.mask[:,:,:T,:k.size(2)])
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
            # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, C)
            y = (attn @ v).transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super(Block,self).__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CasualSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.MLP =MLP(config)

    def forward (self, x, use_cache=False):
        x = x + self.attn(self.ln1(x), use_cache=use_cache)
        x = x + self.MLP(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config)  for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, eps=1e-5),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self.init_weights)#  这是使用 PyTorch 的 apply 方法，递归地调用 self 对象及其所有子模块上的 init_weights 方法。

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self, x, target=None, use_cache=False):
        B, T = x.size()
        assert T <= self.config.block_size, f"Cannot forward, model block size is {self.config.block_size}, but input sequence has length {T}."
        position_ids = torch.arange(T, dtype=torch.long, device=x.device)
        x = self.transformer.wte(x) + self.transformer.wpe(position_ids)

        for block in self.transformer.h:
            x = block(x, use_cache=use_cache)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTconfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay,learing_rate,device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_group = {
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        }
        num_params = sum(p.numel() for p in param_dict.values())
        num_nodecay_params = sum(p.numel() for p in no_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_params:,} parameters")
        print(f"num no-decayed parameter tensors: {len(no_decay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_group, lr=learing_rate, betas=(0.9, 0.95), eps=1e-8,fused=use_fused)
        return optimizer

