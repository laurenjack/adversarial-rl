class ModelConfig:
    def __init__(
        self,
        id: str,
        vocab: int,
        layers: int,
        max_length: int,
        h: int,
        q_heads: int,
        kv_heads: int,
        hidden: int,
        epsilon: float,
    ):
        self.id = id
        self.vocab = vocab
        self.layers = layers
        self.max_length = max_length
        self.h = h
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.hidden = hidden
        self.epsilon = epsilon
        self.head_size = h // q_heads 