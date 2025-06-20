import torch
from torch import nn


VOCAB_SIZE = 32000
NUM_LAYERS = 32
MAX_CONTEXT_LENGTH = 4096
H = 4096
ATTENTION_HEADS = 32
HEAD_SIZE = H // ATTENTION_HEADS
MLP_HIDDEN = 11008
EPSILON = 1e-6


class RmsNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + EPSILON)
        return self.weight * x / norm


class KvCache:

    def __init__(self, b):
        self.k = torch.zeros(b, ATTENTION_HEADS, MAX_CONTEXT_LENGTH, HEAD_SIZE, dtype=torch.bfloat16)
        self.v = torch.zeros(b, ATTENTION_HEADS, MAX_CONTEXT_LENGTH, HEAD_SIZE, dtype=torch.bfloat16)


class SelfAttention(nn.Module):

    def __init__(self, full_mask, sin, cos):
        super().__init__()
        self.full_mask = full_mask
        self.sin = sin
        self.cos = cos
        self.q_proj = nn.Linear(H, H, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(H, H, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(H, H, bias=False, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(H, H, bias=False, dtype=torch.bfloat16)
        # Hacky was to do caching of KV but keeps things simple, this is set explicitly in the outer rollout loop
        self.cache = None

    def _rotate(self, x, start):
        # Apply the ROPE matrix multiplication as an elementwise product
        s = x.shape[2]
        sin = self.sin[start:s + start, :]
        cos = self.cos[start:s + start, :]
        even, odd = x[..., ::2], x[..., 1::2]
        # Every position is * by cos, odd positions get -sin, even gets +sin, achieving the HEAD_SIZE // 2 rotations
        out_even = cos * even - sin * odd
        out_odd = sin * even + cos * odd
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

    def forward(self, x, start=0):
        b, s, _ = x.shape
        # Compute matrix multiplications for attention and reshape so that each head has its own index in the 2nd dim
        Q, K, V = [w(x).view(b, s, ATTENTION_HEADS, HEAD_SIZE) for w in [self.q_proj, self.k_proj, self.v_proj]]
        # Now move the ATTENTION_HEADS dim so that it is treated as a batch dimension in the matrix multiplications
        Q, K, V = [T.transpose(1, 2) for T in [Q, K, V]]
        # K, Q and V now have shape [b, ATTENTION_HEADS, s, HEAD_SIZE]

        # Apply ROPE position rotation
        Q = self._rotate(Q, start)
        K = self._rotate(K, start)
        # A non-zero start indicates the programmer wants to use the cache up to start
        if self.cache:
            self.cache.k = self.cache.k.to(x)
            self.cache.v = self.cache.v.to(x)
            self.cache.k[:, :, start:start+s, :] = K
            self.cache.v[:, :, start:start+s, :] = V
            K = self.cache.k[:, :, :start+s, :]
            V = self.cache.v[:, :, :start+s, :]

        z = Q @ K.transpose(2, 3) / HEAD_SIZE ** 0.5  # [b, ATTENTION_HEADS, s, start + s]
        mask = self.full_mask[:s, :start + s]
        # neg_inf = torch.finfo(z.dtype).min
        z = z.masked_fill(~mask, -1e6)
        A = torch.softmax(z, dim=-1)
        context = A @ V   # [b, ATTENTION_HEADS, s, HEAD_SIZE]

        # Switch the shape back to a stack of heads i.e. [b, s, H]
        context = context.transpose(1, 2).reshape(b, s, H)
        return self.out_proj(context)


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, MLP_HIDDEN, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(H, MLP_HIDDEN, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(MLP_HIDDEN, H, bias=False, dtype=torch.bfloat16)
        self.silu = nn.SiLU()

    def forward(self, x):
        z_gate = self.gate_proj(x)
        z = self.up_proj(x)
        hidden = self.silu(z_gate) * z
        return self.down_proj(hidden)


class LlamaBlock(nn.Module):

    def __init__(self, full_mask, sin, cos):
        super().__init__()
        self.input_layernorm = RmsNorm()
        self.attention = SelfAttention(full_mask, sin, cos)
        self.post_attention_layernorm = RmsNorm()
        self.mlp = MLP()

    def forward(self, x, start=0):
        #  x: [b, s, H]
        normed = self.input_layernorm(x)
        x = x + self.attention(normed, start=start)
        normed = self.post_attention_layernorm(x)
        return x + self.mlp(normed)

       
class LlamaCode2(nn.Module):

    def __init__(self):
        super().__init__()
        # Lower triangular bbut the ecause we can only look to the current token and backwards
        full_mask = torch.tril(torch.ones((MAX_CONTEXT_LENGTH, MAX_CONTEXT_LENGTH))).bool()
        self.register_buffer("full_mask", full_mask, persistent=False)
        # Generate the rope rotator for positional encoding
        sin, cos = _rope_vectors()
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

        # Define the model
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, H, dtype=torch.bfloat16)
        self.blocks = nn.ModuleList([LlamaBlock(full_mask, sin, cos) for _ in range(NUM_LAYERS)])
        self.norm = RmsNorm()
        self.lm_head = nn.Linear(H, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, token_indices, start=0):
        x = self.embed_tokens(token_indices)  # token_indices has shape [b, s],  x has shape [b, s, H]
        for block in self.blocks:
            x = block(x, start=start)
        x = self.norm(x)
        return self.lm_head(x)


def _rope_vectors():
    # Create the angles for rotation
    token_index = torch.arange(MAX_CONTEXT_LENGTH, dtype=torch.bfloat16)
    d_period = [10000 ** (-2 * i / HEAD_SIZE) for i in range(HEAD_SIZE // 2)]
    d_period = torch.tensor(d_period, dtype=torch.bfloat16)
    angles = torch.outer(token_index, d_period)  # [s, HEAD_SIZE // 2]
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return sin, cos






