import torch
from torch import nn

# Import shared model utilities and constants
import rl.model.base as base
from rl.model.base import Model
from rl.config import ModelConfig

# Global config for Llama2-Code
LLAMA2_CONFIG = ModelConfig(
    id="meta-llama/CodeLlama-7b-Instruct-hf",
    vocab=32016,
    layers=32,
    max_length=4096,
    h=4096,
    q_heads=32,
    kv_heads=32,
    hidden=11008,
    epsilon=1e-6,
)

class RmsNorm(nn.Module):

    def __init__(self, mc):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(mc.h, dtype=torch.bfloat16))
        self.epsilon = mc.epsilon

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)
        return self.weight * x / norm


class KvCache:

    def __init__(self, b, mc):
        self.k = torch.zeros(b, mc.kv_heads, mc.max_length, mc.head_size, dtype=torch.bfloat16)
        self.v = torch.zeros(b, mc.kv_heads, mc.max_length, mc.head_size, dtype=torch.bfloat16)


class SelfAttention(nn.Module):

    def __init__(self, full_mask, mc, sin, cos):
        super().__init__()
        self.full_mask = full_mask
        self.sin = sin
        self.cos = cos
        self.mc = mc
        self.q_proj = nn.Linear(mc.h, mc.h, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(mc.h, mc.kv_heads * mc.head_size, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(mc.h, mc.kv_heads * mc.head_size, bias=False, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(mc.h, mc.h, bias=False, dtype=torch.bfloat16)
        # Hacky way to do caching of KV but keeps things simple, this is set explicitly in the outer rollout loop
        self.cache = None

    def set_kv_cache(self, b: int):
        self.cache = KvCache(b, self.mc)

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        group = self.mc.q_heads // self.mc.kv_heads
        b, _, s, _ = x.shape
        return (
            x.unsqueeze(2)
             .expand(-1, -1, group, -1, -1)
             .reshape(b, self.mc.q_heads, s, self.mc.head_size)
        )

    def rotate(self, x: torch.Tensor, start: int) -> torch.Tensor:
        s = x.shape[2]
        sin = self.sin[start : s + start, :]
        cos = self.cos[start : s + start, :]
        even, odd = x[..., ::2], x[..., 1::2]
        out_even = cos * even - sin * odd
        out_odd = sin * even + cos * odd
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

    def forward(self, x, attention_mask=None, start=0):
        b, s, _ = x.shape
        Q = self.q_proj(x).view(b, s, self.mc.q_heads, self.mc.head_size)
        K = self.k_proj(x).view(b, s, self.mc.kv_heads, self.mc.head_size)
        V = self.v_proj(x).view(b, s, self.mc.kv_heads, self.mc.head_size)
        Q, K, V = [T.transpose(1, 2) for T in [Q, K, V]]
        # K, Q and V now have shape [b, heads, s, head_size]

        # Apply ROPE position rotation
        Q = self.rotate(Q, start)
        K = self.rotate(K, start)
        # A non-zero start indicates the programmer wants to use the cache up to start
        if self.cache:
            self.cache.k = self.cache.k.to(x)
            self.cache.v = self.cache.v.to(x)
            self.cache.k[:, :, start:start+s, :] = K
            self.cache.v[:, :, start:start+s, :] = V
            K = self.cache.k[:, :, :start+s, :]
            V = self.cache.v[:, :, :start+s, :]
        
        # Expand the KV cache to match the query head count
        if self.mc.q_heads != self.mc.kv_heads:
            K = self.expand(K)
            V = self.expand(V)
        z = Q @ K.transpose(2, 3) / self.mc.head_size ** 0.5  # [b, q_head, s, start + s]
        mask = self.full_mask[:s, :start + s] .unsqueeze(0).unsqueeze(0)  # [1, 1, s, start + s]
        if attention_mask is not None:
            attn = attention_mask[:, :start + s].to(torch.bool)
            attn = attn.unsqueeze(1).unsqueeze(2)
            mask = mask & attn  # (b, 1, s, start + s)
        z = z.masked_fill(~mask, -1e6)
        A = torch.softmax(z, dim=-1)
        context = A @ V   # [b, q_head, s, head_size]
        if attention_mask is not None:
            q_mask = attention_mask[:, :s].unsqueeze(1).unsqueeze(-1)  # (b,1,s,1)
            context = context * q_mask
        # Switch the shape back to a stack of heads i.e. [b, s, H]
        context = context.transpose(1, 2).reshape(b, s, self.mc.h)
        return self.out_proj(context)


class MLP(nn.Module):

    def __init__(self, mc):
        super().__init__()
        self.gate_proj = nn.Linear(mc.h, mc.hidden, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(mc.h, mc.hidden, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(mc.hidden, mc.h, bias=False, dtype=torch.bfloat16)
        self.silu = nn.SiLU()

    def forward(self, x):
        z_gate = self.gate_proj(x)
        z = self.up_proj(x)
        hidden = self.silu(z_gate) * z
        return self.down_proj(hidden)


class LlamaBlock(nn.Module):

    def __init__(self, full_mask, mc, sin, cos):
        super().__init__()
        self.input_layernorm = RmsNorm(mc)
        self.attention = SelfAttention(full_mask, mc, sin, cos)
        self.post_attention_layernorm = RmsNorm(mc)
        self.mlp = MLP(mc)

    def forward(self, x, attention_mask=None, start=0):
        #  x: [b, s, H]
        normed = self.input_layernorm(x)
        x = x + self.attention(normed, attention_mask=attention_mask, start=start)
        normed = self.post_attention_layernorm(x)
        return x + self.mlp(normed)
    
    def set_kv_cache(self, b: int):
        self.attention.set_kv_cache(b)

       

class Llama2Code(Model):
    """Implementation of Meta's 7B *Code Llama* model in pure PyTorch."""

    def __init__(self):
        super().__init__(LLAMA2_CONFIG)
        # Lower triangular because we can only look to the current token and backwards
        full_mask = torch.tril(torch.ones((self.max_length, self.max_length))).bool()
        self.register_buffer("full_mask", full_mask, persistent=False)

        # Define the model
        self.embed_tokens = nn.Embedding(self.vocab, self.h, dtype=torch.bfloat16)
        self.blocks = nn.ModuleList([LlamaBlock(full_mask, self.mc, self.sin, self.cos) for _ in range(self.layers)])
        self.norm = RmsNorm(self.mc)
        self.lm_head = nn.Linear(self.h, self.vocab, bias=False)

    def set_kv_cache(self, b: int):
        for block in self.blocks:
            block.set_kv_cache(b)

    def forward(self, token_indices, attention_mask=None, start=0):
        if attention_mask is None:
            attention_mask = torch.ones_like(token_indices, dtype=torch.bool)

        x = self.embed_tokens(token_indices)  # [b, s, H]
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, start=start)
        x = self.norm(x)
        return self.lm_head(x)
