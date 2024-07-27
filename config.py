from dataclasses import dataclass
import torch

@dataclass
class GPT_CONFIG:
    out_dir: str = './out'
    eval_interval: int = 10
    save_model: bool = True

    # train config
    batch_size: int = 12
    block_size: int = 1024
    gradient_accumulation_steps = 40

    # model config
    n_layer: int = 12
    n_head: int = 12
    embedding_depth: int = 768
    bias: bool = False
    dropout = 0.0


    # optimizer config
    lr: float = 6e-4
    iter: int = 100
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    gradient_clipping: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    #---
    dtype: str = 'bfloat16'
    compile: bool = True
    seed_offset: int = 0
    tokens_per_iter: int = 12 * 1024  # batch_size * block_size
    torch_16bit_allow: bool = True
    torch_allow_tf32: bool = True
    log_interval = 1

    def __str__(self):
        config_str = (
            f"GPT Configuration:\n"
            f"Output Directory: {self.out_dir}\n"
            f"Evaluation Interval: {self.eval_interval}\n"
            f"Save Model: {self.save_model}\n"
            f"\nTrain Config:\n"
            f"Batch Size: {self.batch_size}\n"
            f"Block Size: {self.block_size}\n"
            f"\nModel Config:\n"
            f"Number of Layers: {self.n_layer}\n"
            f"Number of Heads: {self.n_head}\n"
            f"Embedding Depth: {self.embedding_depth}\n"
            f"Bias: {self.bias}\n"
            f"\nOptimizer Config:\n"
            f"Learning Rate: {self.lr}\n"
            f"Iterations: {self.iter}\n"
            f"Weight Decay: {self.weight_decay}\n"
            f"Beta1: {self.beta1}\n"
            f"Beta2: {self.beta2}\n"
            f"Gradient Clipping: {self.gradient_clipping}\n"
            f"Device: {self.device}\n"
            f"\nMiscellaneous:\n"
            f"Data Type: {self.dtype}\n"
            f"Compile: {self.compile}\n"
            f"Seed Offset: {self.seed_offset}\n"
            f"Tokens per Iteration: {self.tokens_per_iter}\n"
            f"Allow Torch 16-bit: {self.torch_16bit_allow}\n"
            f"Allow Torch TF32: {self.torch_allow_tf32}\n"
        )
        return config_str


