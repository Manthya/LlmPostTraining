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
| `--use_kl_regularization` | `False` | KL regularization (requires 2x memory) |
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

## Training Results

Results from training on OpenThoughts-114k with Qwen/Qwen2.5-0.5B:

```
EVALUATION RESULTS SUMMARY
============================================================

📊 Base Model (Qwen/Qwen2.5-0.5B):
   - Avg Reasoning Quality Score: 0.260
   - % with Reasoning: 90.0%
   - Avg Generation Time: 4.13s

📊 SFT Model (./outputs/distillation/run_20260204_174941/final_model):
   - Avg Reasoning Quality Score: 0.280
   - % with Reasoning: 100.0%
   - Avg Generation Time: 18.95s

📈 Improvement (SFT vs Base):
   - Reasoning Quality Delta: +0.020
   - Reasoning % Delta: +10.0%
```

**Observations:**
- SFT model consistently produces reasoning with `<think>` tags (100% vs 90%)
- Reasoning quality improvement is modest with limited samples
- Larger improvements expected with more training data and epochs

## Benchmark Evaluation

For rigorous evaluation on standard benchmarks, use `run_benchmark_evaluation.py`:

### Supported Benchmarks

| Benchmark | Description | Test Size |
|-----------|-------------|-----------|
| **GSM8K** | Grade school math word problems | 1,319 |
| **MATH** | Competition-level math | 5,000 |
| **ARC-Challenge** | AI2 science reasoning | 1,172 |

### Running Benchmarks

```bash
# Quick test on GSM8K (100 samples)
python run_benchmark_evaluation.py \
    --base_model Qwen/Qwen2.5-0.5B \
    --sft_model ./outputs/distillation/run_XXXXXX/final_model \
    --benchmarks gsm8k \
    --max_samples 100

# Full evaluation on GSM8K (all 1,319 examples)
python run_benchmark_evaluation.py \
    --base_model Qwen/Qwen2.5-0.5B \
    --sft_model ./outputs/distillation/run_XXXXXX/final_model \
    --benchmarks gsm8k

# Full evaluation on all benchmarks
python run_benchmark_evaluation.py \
    --base_model Qwen/Qwen2.5-0.5B \
    --sft_model ./outputs/distillation/run_XXXXXX/final_model \
    --benchmarks gsm8k math arc_challenge
```

### Benchmark Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--base_model` | Required | Base model to evaluate |
| `--sft_model` | Required | Path to SFT model |
| `--benchmarks` | `gsm8k` | Benchmarks to run |
| `--max_samples` | All | Samples per benchmark |
| `--output_dir` | `./outputs/benchmarks` | Results directory |

### Example Output

```
📊 FINAL EVALUATION COMPARISON REPORT
======================================================================

Benchmark               Base Pass@1      SFT Pass@1          Δ
------------------------------------------------------------
gsm8k                       12.50%          18.75%     +6.25%
math                         5.00%           8.00%     +3.00%
arc_challenge               35.00%          42.00%     +7.00%
------------------------------------------------------------
OVERALL                     17.50%          22.92%     +5.42%
```

### Evaluation Methodology

Following **DeepSeek-R1** evaluation approach:
- Temperature: 0.6, Top-p: 0.95
- Pass@1: Single sample accuracy
- Answer extraction from model responses
- Exact match comparison with ground truth

## Files

| File | Description |
|------|-------------|
| `run_distillation_pipeline.py` | Main pipeline script |
| `config.py` | Configuration dataclasses |
| `data_preprocessing.py` | Dataset loading and preprocessing |
| `sft_trainer.py` | SFT training implementation |
| `evaluation.py` | Model evaluation and comparison |
| `run_benchmark_evaluation.py` | **NEW:** Benchmark evaluation (GSM8K, MATH, ARC) |
| `requirements.txt` | Python dependencies |

## References

1. [DeepSeek-R1 Paper](https://arxiv.org/abs/2401.02954) - DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
2. [OpenThoughts-114k Dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
3. [PEFT Library](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning

## License

MIT License - See main repository LICENSE file.
