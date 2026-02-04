"""
Configuration for Reasoning Distillation Pipeline
Following DeepSeek-R1-Distill methodology

LEARNINGS APPLIED FROM SFT EXPERIMENTS:
1. Single consistent template (no template randomization)
2. Lower learning rate (2e-5 with cosine decay)
3. KL regularization to prevent catastrophic forgetting
4. Quality data filtering over quantity
5. Higher warmup ratio for stability
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class DistillationConfig:
    """Configuration for the distillation pipeline"""
    
    # Model configuration
    base_model_name: str = "Qwen/Qwen2.5-0.5B"  # Non-reasoning base model
    model_max_length: int = 1024  # Reduced from 2048 for memory efficiency
    
    # Dataset configuration
    dataset_name: str = "open-thoughts/OpenThoughts-114k"
    dataset_split: str = "train"
    max_samples: Optional[int] = None  # None for full dataset
    validation_split: float = 0.05
    
    # Training configuration (OPTIMIZED FROM EXPERIMENTS + MEMORY)
    output_dir: str = "./outputs/distillation"
    num_train_epochs: int = 2  # Reduced: smaller models overfit faster
    per_device_train_batch_size: int = 1  # Reduced for memory
    per_device_eval_batch_size: int = 1  # Reduced for memory
    gradient_accumulation_steps: int = 16  # Increased to compensate for smaller batch
    learning_rate: float = 2e-5  # CRITICAL: Low LR preserves base knowledge
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1  # 10% warmup for stability
    lr_scheduler_type: str = "cosine"  # Cosine decay works best
    max_grad_norm: float = 1.0  # Gradient clipping for stability
    gradient_checkpointing: bool = True  # MEMORY: Trade compute for memory
    
    # KL Regularization (LEARNED FROM INSTRUCTGPT EXPERIMENTS)
    # NOTE: Disabled by default - requires 2x memory (reference model)
    # Enable only on systems with sufficient VRAM (16GB+)
    use_kl_regularization: bool = False  # Prevents catastrophic forgetting
    kl_coef: float = 0.1  # KL penalty coefficient (β)
    
    # LoRA configuration (for efficient training)
    use_lora: bool = True
    lora_r: int = 16  # Reduced from 64 for memory efficiency
    lora_alpha: int = 32  # Reduced proportionally
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 500
    
    # Early stopping (PREVENTS OVERFITTING)
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    # Device configuration
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    fp16: bool = field(default_factory=lambda: torch.cuda.is_available())
    bf16: bool = False
    
    # Seed for reproducibility
    seed: int = 42
    
    # Evaluation configuration
    eval_reasoning_tasks: List[str] = field(default_factory=lambda: [
        "gsm8k",
        "math",
        "logic"
    ])
    num_eval_samples: int = 100
    
    def __post_init__(self):
        """Validate and adjust configuration"""
        if self.device == "cpu":
            self.fp16 = False
            self.bf16 = False
            # Reduce batch size for CPU
            self.per_device_train_batch_size = 1
            self.per_device_eval_batch_size = 1
            # Disable KL regularization on CPU (memory intensive)
            self.use_kl_regularization = False


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    
    # Model paths
    base_model_path: str = "Qwen/Qwen2.5-0.5B"
    sft_model_path: str = "./outputs/distillation/final_model"
    
    # Evaluation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Output
    results_output_path: str = "./outputs/distillation/evaluation_results.json"
