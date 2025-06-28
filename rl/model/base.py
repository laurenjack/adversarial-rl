import torch
from torch import nn

from rl.config import ModelConfig

class Model(nn.Module):
    """Shared base-class for all language models used in this repository.

    """
    def __init__(self, mc: ModelConfig):
        super().__init__()
        self.mc = mc
        self.id = mc.id
        self.vocab = mc.vocab
        self.layers = mc.layers
        self.max_length = mc.max_length
        self.h = mc.h
        self.q_heads = mc.q_heads
        self.kv_heads = mc.kv_heads
        self.hidden = mc.hidden
        self.epsilon = mc.epsilon
        self.head_size = mc.head_size
        sin, cos = self._init_rope_vectors()
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        self.sin = sin
        self.cos = cos

    def _init_rope_vectors(self):
        token_index = torch.arange(self.max_length, dtype=torch.bfloat16)
        d_period = torch.tensor(
            [10000 ** (-2 * i / self.head_size) for i in range(self.head_size // 2)],
            dtype=torch.bfloat16,
        )
        angles = torch.outer(token_index, d_period)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return sin, cos
