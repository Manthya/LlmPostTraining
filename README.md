# 🚀 LLM Post-Training Experiments

A comprehensive repository for experimenting with different stages of LLM post-training, from basic SFT to advanced LoRA/QLoRA techniques.

## 📋 Overview

This repository implements a **3-stage progressive approach** to LLM post-training:

| Stage | Name | Goal | Key Learning |
|-------|------|------|--------------|
| 1 | **Normal SFT** | Teach task completion | Output structure, basic patterns |
| 2 | **Instruction Tuning** | Teach instruction following | Paraphrase robustness, multi-format |
| 3 | **LoRA/QLoRA** | Scale efficiently | Memory-efficient training |

## 🏗️ Repository Structure

```
LlmPostTraining/
├── configs/                        # Configuration files
│   ├── base_config.yaml           # Base configuration
│   ├── stage1_sft_config.yaml     # Stage 1 settings
│   ├── stage2_instruction_config.yaml  # Stage 2 settings
│   └── stage3_lora_config.yaml    # Stage 3 settings
│
├── notebooks/                      # Training notebooks
│   ├── 01_stage1_normal_sft.ipynb      # Stage 1: Normal SFT
│   ├── 02_stage2_instruction_tuning.ipynb  # Stage 2: Instruction Tuning
│   ├── 03_stage3_lora_qlora.ipynb      # Stage 3: LoRA/QLoRA
│   └── 04_evaluation_compare_stages.ipynb  # Comprehensive evaluation
│
├── src/                           # Source code
│   ├── __init__.py               # Package exports
│   ├── data_utils.py             # Data loading & formatting
│   ├── training_utils.py         # Training utilities
│   ├── test_queries.py           # Evaluation test queries
│   └── evaluation.py             # Evaluation framework
│
├── data/                          # Training data (generated)
│   ├── stage1/                   # Stage 1 data
│   ├── stage2/                   # Stage 2 data
│   └── evaluation/               # Evaluation data
│
├── outputs/                       # Training outputs (generated)
│   ├── stage1_sft/               # Stage 1 checkpoints
│   ├── stage2_instruction/       # Stage 2 checkpoints
│   ├── stage3_lora/              # Stage 3 adapters
│   └── evaluation/               # Evaluation results
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/LlmPostTraining.git
cd LlmPostTraining

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training Notebooks

Open the notebooks in order:

1. **Stage 1**: `notebooks/01_stage1_normal_sft.ipynb`
2. **Stage 2**: `notebooks/02_stage2_instruction_tuning.ipynb`
3. **Stage 3**: `notebooks/03_stage3_lora_qlora.ipynb`
4. **Evaluation**: `notebooks/04_evaluation_compare_stages.ipynb`

## 📚 Stage Details

### Stage 1: Normal SFT (Prompt → Output)

> "Teach the model to complete correctly"

**What it does:**
- Learns output structure
- Learns task framing
- Minimal behavior shaping

**Data Format:**
```json
{
  "prompt": "Question or task",
  "output": "Correct completion"
}
```

**Expected Behavior:**
- ✅ Loss decreases smoothly
- ✅ Outputs are more task-correct
- ❌ No "chat intelligence" yet
- ⚠️ May overfit to specific phrasings (this is expected!)

---

### Stage 2: Instruction Tuning (Instruction → Response)

> "Teach the model how to be told what to do"

**What it adds:**
- Instruction abstraction
- Robustness to paraphrasing
- Multi-task generalization

**Data Format:**
```json
{
  "instruction": "What to do",
  "input": "Optional context",
  "response": "The answer"
}
```

**Key Technique: Template Randomization**

We use multiple prompt templates to prevent instruction overfitting:

```python
TEMPLATES = [
    "### Instruction:\n{instruction}\n\n### Response:",
    "Instruction: {instruction}\n\nResponse:",
    "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant",
]
```

**Expected Behavior:**
- ✅ Same task works across multiple phrasings
- ✅ General instruction-following improves
- ⚠️ May still hallucinate (needs RLHF for full fix)

---

### Stage 3: LoRA/QLoRA (Memory-Efficient SFT)

> "Scale without breaking the model"

**Why this stage exists:**
- Full fine-tuning doesn't scale
- Risk of catastrophic forgetting
- Cost explodes with model size

**Techniques:**
- **LoRA**: Low-Rank Adaptation - train small adapter matrices
- **QLoRA**: 4-bit quantization + LoRA adapters

**Efficiency Comparison:**

| Metric | Full Fine-tune | LoRA | QLoRA |
|--------|---------------|------|-------|
| Memory | 100% | ~10% | ~5% |
| Trainable Params | 100% | ~2% | ~2% |
| Training Speed | Slow | Fast | Fast |
| Quality | Baseline | Similar | Similar |

**Expected Behavior:**
- ✅ Memory usage significantly reduced
- ✅ Similar performance to full fine-tuning
- ✅ Base model knowledge preserved
- ✅ Tiny adapter files for deployment

---

## 🧪 Evaluation Framework

The evaluation framework tests models across multiple dimensions:

| Category | What it Tests |
|----------|---------------|
| **Factual Knowledge** | Basic knowledge retention |
| **Reasoning** | Logical reasoning abilities |
| **Instruction Following** | Ability to follow specific constraints |
| **Creative Tasks** | Creative generation quality |
| **Code Generation** | Coding abilities |
| **Paraphrase Robustness** | Same task, different phrasing |

### Test Queries

Located in `src/test_queries.py`, includes:
- 30+ test queries across categories
- Multiple paraphrase variants per query
- Expected behavior annotations per stage

---

## 🔧 Base Model

We use **Qwen2.5-1.5B** as the base model because:

1. **Not heavily post-trained** - Good for seeing training effects
2. **Small enough for experiments** - Runs on consumer GPUs
3. **Good architecture** - Modern transformer with quality tokenizer

**Alternatives:**
- `microsoft/phi-2` (2.7B)
- `mistralai/Mistral-7B-v0.1` (if you have more compute)

---

## 💡 Key Insights

### What Each Stage Teaches

1. **Stage 1** proves that basic SFT works but creates "memorizers" not "understanders"
2. **Stage 2** shows that instruction abstraction requires diverse training formats
3. **Stage 3** demonstrates that you don't need to train all parameters

### Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Stage 1 overfits to phrasing | This is expected - move to Stage 2 |
| Stage 2 overfits to one template | Use template randomization |
| LoRA adapters too small (low r) | Increase rank (r=64+ for complex tasks) |
| Model forgets base knowledge | Use LoRA instead of full fine-tuning |

---

## 📊 Hardware Requirements

| Stage | Minimum GPU | Recommended |
|-------|-------------|-------------|
| Stage 1 | 16GB VRAM | 24GB VRAM |
| Stage 2 | 16GB VRAM | 24GB VRAM |
| Stage 3 (QLoRA) | **8GB VRAM** | 16GB VRAM |

---

## 🤝 Contributing

Contributions welcome! Areas to explore:

- [ ] Add more test queries
- [ ] Implement RLHF (Stage 4)
- [ ] Add DPO training
- [ ] Multi-GPU training support
- [ ] More evaluation metrics

---

## 📖 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca)
- [PEFT Library](https://github.com/huggingface/peft)

---

## 📝 License

MIT License - feel free to use for your experiments!

---

**Happy Training! 🎓**