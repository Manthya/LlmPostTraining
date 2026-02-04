# Reasoning Distillation Pipeline

## Overview

This module implements reasoning distillation following the **DeepSeek-R1-Distill** paper methodology. The goal is to transfer reasoning capabilities from reasoning-enhanced training data (OpenThoughts-114k) to a non-reasoning base model using pure Supervised Fine-Tuning (SFT).

## Method

### DeepSeek-R1-Distill Approach

The DeepSeek-R1-Distill paper demonstrates that reasoning capabilities can be distilled from a large reasoning model to smaller models through:

1. **Chain-of-Thought Data**: Using training data that contains explicit reasoning steps wrapped in `<think>` tags
2. **Pure SFT**: Simple supervised fine-tuning on this reasoning-enhanced data
3. **Format Learning**: Teaching the model to produce structured reasoning before final answers

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────┐
│                    DISTILLATION PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1: Data Preparation                                  │
│  ├── Load OpenThoughts-114k dataset                        │
│  ├── Preprocess with <think> tag structure                  │
│  └── Format for chat template                               │
│                                                              │
│  Stage 2: SFT Training                                      │
│  ├── Load non-reasoning base model (Qwen/Llama)            │
│  ├── Apply LoRA for efficient training                      │
│  └── Train on reasoning data                                │
│                                                              │
│  Stage 3: Evaluation                                        │
│  ├── Compare base vs SFT model                              │
│  ├── Measure reasoning quality                              │
│  └── Generate comparison report                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Dataset

### OpenThoughts-114k

The pipeline uses the [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) dataset, which contains:

- **114k+ reasoning examples** covering math, logic, and analytical problems
- **Chain-of-thought reasoning** with explicit step-by-step solutions
- **`<think>` tag structure** for separating reasoning from final answers

### Data Format

```
<|im_start|>system
You are a helpful assistant that thinks step by step...
<|im_start|>user
{problem}
<|im_start|>assistant
<think>
{step-by-step reasoning}
</think>

{final answer}
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install torch transformers datasets peft accelerate trl
```

## Usage

### Quick Start

```bash
# Run full pipeline (train + evaluate)
python run_distillation_pipeline.py --mode all

# Quick test with small sample
python run_distillation_pipeline.py --mode all --max_samples 100 --epochs 1

# Recommended: With KL regularization (prevents catastrophic forgetting)
python run_distillation_pipeline.py --mode all --max_samples 1000 --use_kl_regularization
```

### Training Only

```bash
python run_distillation_pipeline.py --mode train \
    --model Qwen/Qwen2.5-0.5B \
    --epochs 3 \
    --max_samples 10000
```

### Evaluation Only

```bash
python run_distillation_pipeline.py --mode evaluate \
    --model Qwen/Qwen2.5-0.5B \
    --sft_model_path ./outputs/distillation/final_model
```

### Full Options

```bash
python run_distillation_pipeline.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `all` | Pipeline mode: `all`, `train`, `evaluate` |
| `--model` | `Qwen/Qwen2.5-0.5B` | Base model name/path |
| `--dataset` | `open-thoughts/OpenThoughts-114k` | Training dataset |
| `--max_samples` | `None` | Max training samples |
| `--epochs` | `3` | Training epochs |
| `--batch_size` | `2` | Per-device batch size |
| `--gradient_accumulation` | `8` | Gradient accumulation steps |
| `--learning_rate` | `2e-5` | Learning rate |
| `--use_lora` | `True` | Use LoRA for efficient training |
| `--output_dir` | Auto | Output directory |
| `--use_kl_regularization` | `True` | KL regularization to prevent forgetting |
| `--kl_coef` | `0.1` | KL divergence coefficient |
| `--early_stopping_patience` | `3` | Early stopping patience epochs |
| `--max_grad_norm` | `1.0` | Gradient clipping norm |

## Supported Models

The pipeline supports any HuggingFace causal language model. Tested models:

| Model | Size | Recommended For |
|-------|------|-----------------|
| `Qwen/Qwen2.5-0.5B` | 0.5B | Quick experiments, CPU testing |
| `Qwen/Qwen2.5-1.5B` | 1.5B | Good balance of speed and quality |
| `Qwen/Qwen2.5-7B` | 7B | Best quality, requires GPU |
| `meta-llama/Llama-2-7b-hf` | 7B | Alternative base model |
| `meta-llama/Llama-3.1-8B` | 8B | Latest Llama model |

## Hardware Requirements

### Minimum (CPU)
- 16GB RAM
- Training will be slow (use `--max_samples 100` for testing)

### Recommended (GPU)
- NVIDIA GPU with 8GB+ VRAM
- 32GB RAM
- CUDA 11.8+

### Optimal
- NVIDIA A100/H100 with 40GB+ VRAM
- 64GB RAM
- For full dataset training

## Output Structure

```
outputs/distillation/run_YYYYMMDD_HHMMSS/
├── config.json                 # Training configuration
├── final_model/               # Trained LoRA adapters
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer.json
├── merged_model/              # Full merged model
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
├── evaluation_results.json    # Comparison results
└── distillation_pipeline.log  # Training logs
```

## Evaluation Metrics

The evaluation compares base and SFT models on:

| Metric | Description |
|--------|-------------|
| **Reasoning Quality Score** | 0-1 score based on structure, length, steps |
| **% with Reasoning** | Percentage of responses with reasoning |
| **Has Think Tags** | Whether model uses proper `<think>` structure |
| **Step Count** | Number of reasoning steps detected |
| **Generation Time** | Time to generate response |

## Example Results

```
EVALUATION RESULTS SUMMARY
============================================================

📊 Base Model (Qwen/Qwen2.5-0.5B):
   - Avg Reasoning Quality Score: 0.150
   - % with Reasoning: 30.0%
   - Avg Generation Time: 2.45s

📊 SFT Model (./outputs/distillation/final_model):
   - Avg Reasoning Quality Score: 0.650
   - % with Reasoning: 90.0%
   - Avg Generation Time: 3.12s

📈 Improvement (SFT vs Base):
   - Reasoning Quality Delta: +0.500
   - Reasoning % Delta: +60.0%
```

## Files

| File | Description |
|------|-------------|
| `run_distillation_pipeline.py` | Main pipeline script |
| `config.py` | Configuration dataclasses |
| `data_preprocessing.py` | Dataset loading and preprocessing |
| `sft_trainer.py` | SFT training implementation |
| `evaluation.py` | Model evaluation and comparison |
| `requirements.txt` | Python dependencies |

## References

1. [DeepSeek-R1 Paper](https://arxiv.org/abs/2401.02954) - DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
2. [OpenThoughts-114k Dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
3. [PEFT Library](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning

## License

MIT License - See main repository LICENSE file.
