# 🚀 LLM Post-Training: From SFT to RLHF

A hands-on journey through LLM post-training techniques, implementing everything from basic Supervised Fine-Tuning (SFT) to full RLHF with Reward Models and PPO.

> **Base Model**: GPT-2 (124M parameters) - small enough to run on CPU, perfect for learning!

---

## 📋 What This Repository Covers

We implement a **complete 7-notebook learning path** that progressively builds understanding of LLM post-training:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Basic SFT Pipeline (Notebooks 01-04)                          │
│  Learn the fundamentals of fine-tuning and where they fail              │
├─────────────────────────────────────────────────────────────────────────┤
│  PHASE 2: Analysis & InstructGPT-Style Training (Notebooks 05-06)       │
│  Understand WHY basic SFT fails and implement proper fixes              │
├─────────────────────────────────────────────────────────────────────────┤
│  PHASE 3: Full RLHF Pipeline (Notebook 07)                              │
│  Reward Model training + PPO fine-tuning                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Experiments & Results

### Notebook 01: Stage 1 - Normal SFT

**Goal**: Teach GPT-2 to complete tasks using basic supervised fine-tuning.

| Metric | Value |
|--------|-------|
| Dataset | 22 samples (tiny for demo) |
| Training Time | ~11 minutes |
| Final Loss | 3.18 |
| Trainable Params | 124M (100%) |

**Key Findings**:
- ✅ Loss decreases smoothly
- ❌ Model overfits to specific phrasings
- ❌ No generalization to paraphrased questions

---

### Notebook 02: Stage 2 - Instruction Tuning

**Goal**: Improve robustness using multiple instruction templates.

| Metric | Value |
|--------|-------|
| Dataset | 200 Alpaca samples |
| Training Time | ~26 minutes |
| Final Loss | 2.34 |
| Templates Used | 5 different formats |

**Technique: Template Randomization**
```python
TEMPLATES = [
    "### Instruction:\n{instruction}\n\n### Response:",
    "Instruction: {instruction}\n\nResponse:",
    "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant",
]
```

**Key Findings**:
- ✅ Lower loss than Stage 1
- ❌ Outputs became repetitive and degenerate
- ❌ Catastrophic forgetting of base knowledge

---

### Notebook 03: Stage 3 - LoRA Fine-Tuning

**Goal**: Memory-efficient training with Low-Rank Adaptation.

| Metric | Value |
|--------|-------|
| Dataset | 200 Alpaca samples |
| Training Time | ~86 minutes |
| Final Loss | 2.75 |
| Trainable Params | 1.6M (1.29%) |
| LoRA Rank | 16 |

**LoRA Configuration**:
```python
lora_r = 16           # Rank
lora_alpha = 32       # Alpha (2x rank)
target_modules = ["c_attn", "c_proj"]  # GPT-2 attention
```

**Key Findings**:
- ✅ 98.7% fewer trainable parameters
- ✅ Base knowledge better preserved
- ❌ Still showed repetitive outputs

---

### Notebook 04: Evaluation & Comparison

**Goal**: Compare all 3 stages systematically.

| Model | Perplexity | Coherence | Follows Instructions |
|-------|------------|-----------|---------------------|
| GPT-2 Base | Best | High | ❌ No |
| Stage 1 SFT | Medium | Medium | Partially |
| Stage 2 Instruction | Worse | Low | ❌ Degraded |
| Stage 3 LoRA | Medium | Medium | Partially |

**Key Insight**: All 3 stages showed **catastrophic forgetting** and **degenerate outputs**.

---

### Notebook 05: Deep Analysis - Why SFT Failed

**Goal**: Mathematical analysis of what went wrong.

#### The Problem: Distribution Shift

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log P(x_t \mid x_{1:t-1}; \theta)$$

| Issue | Cause | Effect |
|-------|-------|--------|
| **Catastrophic Forgetting** | Full fine-tuning overwrites weights | Lost general knowledge |
| **Distribution Shift** | Training data ≠ pretraining data | Model forgot how to write |
| **Repetition Collapse** | High learning rate + small data | Degenerate loops |
| **Template Conflicts** | 5 different templates | Gradient interference |

**Loss to Perplexity Analysis**:
```
Stage 1: Loss 3.18 → Perplexity 24.1
Stage 2: Loss 2.34 → Perplexity 10.4
Stage 3: Loss 2.75 → Perplexity 15.6
```

Lower loss ≠ better model! Stage 2 had lowest loss but worst outputs.

---

### Notebook 06: InstructGPT-Style SFT

**Goal**: Implement proper SFT based on the InstructGPT paper (Ouyang et al., 2022).

#### InstructGPT Paper Methodology

| Paper Technique | Our Implementation |
|-----------------|-------------------|
| 13,000 demonstrations | 3,000 filtered samples |
| GPT-3 (175B) | GPT-2 (124M) |
| 16 epochs | 2 epochs |
| Pretraining mix | KL regularization |
| Cosine LR schedule | LR = 2e-5 with cosine decay |

**Key Improvements**:
```python
# Single consistent template (no conflicts)
TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"

# KL Regularization to prevent forgetting
kl_loss = kl_divergence(current_logits, original_logits)
total_loss = sft_loss + β * kl_loss  # β = 0.1
```

| Metric | Previous Stages | InstructGPT SFT |
|--------|-----------------|-----------------|
| Template Count | 5 | 1 |
| Learning Rate | 1e-4 to 1e-5 | 2e-5 (cosine) |
| KL Regularization | ❌ None | ✅ β=0.1 |
| Data Quality | Random | Filtered |
| Repetition | High | Low |

**Results**:
- ✅ Non-repetition score: 0.79 (best)
- ✅ Coherent outputs
- ✅ Base knowledge preserved

---

### Notebook 07: RLHF - Reward Model + PPO

**Goal**: Complete the InstructGPT 3-step pipeline with RM and PPO.

#### Step 2: Reward Model Training

| Metric | Value |
|--------|-------|
| Dataset | Anthropic HH-RLHF (5,000 pairs) |
| Architecture | GPT2RewardModel (custom) |
| Training | Binary ranking loss |
| Accuracy | ~65% on held-out data |

**Reward Model Architecture**:
```python
class GPT2RewardModel(nn.Module):
    def __init__(self, model_name):
        self.gpt2 = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids):
        outputs = self.gpt2(input_ids)
        reward = self.reward_head(outputs.last_hidden_state[:, -1])
        return reward
```

#### Step 3: PPO Fine-Tuning

| Metric | Value |
|--------|-------|
| Training Steps | 150 |
| Prompts Used | 300 |
| KL Coefficient | 0.5 (high for stability) |
| Learning Rate | 5e-6 (low to prevent drift) |
| Training Time | ~18 minutes |

**PPO Optimization Journey**:

| Version | Steps | KL Coef | Result |
|---------|-------|---------|--------|
| v1 | 30 | 0.2 | 25% metrics (too much drift) |
| v2 | 100 | 0.2 | Still poor |
| v3 (Final) | 150 | 0.5 | 80% metrics ✅ |

**Final PPO Configuration**:
```python
PPOConfig(
    learning_rate=5e-6,      # Very low
    kl_coef=0.5,             # High penalty
    num_ppo_epochs=2,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
)
```




## 🏗️ Repository Structure

```
LlmPostTraining/
├── notebooks/
│   ├── 01_stage1_normal_sft.ipynb          # Basic SFT
│   ├── 02_stage2_instruction_tuning.ipynb  # Multi-template training
│   ├── 03_stage3_lora_qlora.ipynb          # LoRA fine-tuning
│   ├── 04_evaluation_compare_stages.ipynb  # Stage comparison
│   ├── 05_analysis_and_improvements.ipynb  # Mathematical analysis
│   ├── 06_instruct_tunning_training.ipynb  # InstructGPT SFT
│   └── 07_rlhf_reward_model_ppo.ipynb      # Full RLHF pipeline
│
├── models/
│   └── gpt2/                               # Base GPT-2 model
│
├── outputs/
│   ├── stage1_sft/                         # Stage 1 checkpoint
│   ├── stage2_instruction/                 # Stage 2 checkpoint
│   ├── stage3_lora/                        # LoRA adapters
│   ├── improved_training/                  # InstructGPT SFT model
│   ├── rlhf_training/                      # PPO model + reward model
│   └── evaluation/                         # Results & charts
│
├── src/                                    # Utility modules
├── configs/                                # YAML configurations
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/your-username/LlmPostTraining.git
cd LlmPostTraining

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download Base Model

```bash
# GPT-2 will auto-download, or manually:
huggingface-cli download gpt2 --local-dir ./models/gpt2
```

### 3. Run Notebooks in Order

1. `01_stage1_normal_sft.ipynb` - Learn basic SFT
2. `02_stage2_instruction_tuning.ipynb` - See template randomization
3. `03_stage3_lora_qlora.ipynb` - Try memory-efficient training
4. `04_evaluation_compare_stages.ipynb` - Compare results
5. `05_analysis_and_improvements.ipynb` - Understand failures
6. `06_instruct_tunning_training.ipynb` - Proper InstructGPT SFT
7. `07_rlhf_reward_model_ppo.ipynb` - Full RLHF pipeline

---

## 💡 Key Lessons Learned

### What Works

| Technique | Why It Works |
|-----------|--------------|
| **Single Template** | Avoids gradient conflicts |
| **KL Regularization** | Prevents catastrophic forgetting |
| **Low Learning Rate** | Preserves base knowledge |
| **High KL Penalty in PPO** | Keeps model close to SFT |
| **Quality over Quantity** | 3K good samples > 200 random |

### What Doesn't Work

| Mistake | Consequence |
|---------|-------------|
| Multiple templates | Gradient interference |
| High learning rate | Destroys pretrained weights |
| No regularization | Catastrophic forgetting |
| Too many PPO steps | Model drifts from SFT |
| Tiny datasets | Severe overfitting |

---

## 🎓 What We Learned (Deep Insights)

### 1. Loss ≠ Quality
Our experiments proved that **lower training loss doesn't mean better outputs**:
- Stage 2 had the lowest loss (2.34) but the worst outputs
- The model learned to predict tokens correctly but forgot how to generate coherent text
- **Takeaway**: Always evaluate with generation quality, not just loss

### 2. Catastrophic Forgetting is Real
When fine-tuning GPT-2 on small datasets:
- The model "forgot" its pretrained knowledge
- Outputs became repetitive patterns like `", , , , , ,"`
- **Solution**: Use KL regularization, low learning rate, or LoRA

### 3. Template Consistency Matters
Using 5 different instruction templates caused gradient conflicts:
- Each template tried to pull the model in different directions
- InstructGPT paper uses **single consistent format**
- **Takeaway**: Pick ONE template and stick with it

### 4. PPO is Sensitive to Hyperparameters
Our PPO journey showed:
- Default settings (KL=0.2) caused too much drift → 25% metrics
- Higher KL penalty (0.5) + lower LR (5e-6) → 80% metrics
- **Takeaway**: Start conservative, increase exploration gradually

### 5. Reward Model Quality is Critical
The RM defines what "good" means:
- Biased RM → biased model behavior
- We used Anthropic HH-RLHF (human preferences)
- **Takeaway**: RM training is often more important than PPO tuning

---

## ⚠️ Best Practices for LLM Post-Training

### SFT Best Practices

```python
# ✅ DO
learning_rate = 2e-5          # Low and stable
single_template = True         # Consistent format
warmup_ratio = 0.1             # Gradual warmup
cosine_schedule = True         # Smooth decay

# ❌ DON'T
learning_rate = 1e-4           # Too high
multiple_templates = True      # Causes conflicts
no_warmup = True               # Unstable start
```

### RLHF Best Practices

```python
# ✅ PPO Configuration
kl_coef = 0.5                  # High for stability
learning_rate = 5e-6           # Very low
num_ppo_epochs = 2             # Don't over-train
gradient_clipping = 0.5        # Prevent explosions

# ❌ Common Mistakes
kl_coef = 0.01                 # Too much freedom
learning_rate = 1e-4           # Way too high
training_steps = 1000          # Way too many
```

### Data Quality Checklist

- [ ] Filter out low-quality samples
- [ ] Ensure response length variety
- [ ] Balance task types
- [ ] Remove duplicates and near-duplicates
- [ ] Verify label quality (for preference data)

---

## 📈 Training Metrics Summary

| Notebook | Method | Loss | Time | Key Result |
|----------|--------|------|------|------------|
| 01 | SFT | 3.18 | 11 min | Basic completion |
| 02 | Instruction | 2.34 | 26 min | Repetitive outputs |
| 03 | LoRA | 2.75 | 86 min | Fewer params, similar issues |
| 06 | InstructGPT SFT | Lower | 15 min | Non-repetitive ✅ |
| 07 | PPO | - | 18 min | 80% on metrics ✅ |

---

## 🔮 Future Scope & Latest Innovations

### What We Plan to Explore

- [ ] **DPO (Direct Preference Optimization)** - Simpler than PPO, no reward model needed
- [ ] **IPO (Identity Preference Optimization)** - More robust, prevents overfitting
- [ ] **KTO (Kahneman-Tversky Optimization)** - Works with thumbs up/down data
- [ ] **GRPO (Group Relative Policy Optimization)** - DeepSeek's memory-efficient PPO
- [ ] **Constitutional AI** - Self-improvement without human labels
- [ ] **Reasoning Models** - Train models to "think" step-by-step

### 🌟 Latest Industry Innovations (2025-2026)

| Innovation | Source | Description |
|------------|--------|-------------|
| **DPO** | Rafailov et al. | Bypass reward model entirely, use preference pairs directly |
| **GRPO** | DeepSeek | Memory-efficient PPO variant, used in DeepSeek-R1 |
| **Open-R1** | HuggingFace | Open reproduction of DeepSeek reasoning pipeline |
| **Reasoning RL** | DeepSeek-R1 | Pure RL to teach reasoning without SFT |
| **Agentic RL** | LinkedIn | Multi-step tool use with RL optimization |

### DPO vs PPO: The Future Direction

| Aspect | PPO (RLHF) | DPO |
|--------|------------|-----|
| Complexity | High (RM + RL) | Low (single loss) |
| Stability | Tricky to tune | More stable |
| Data | Needs RM training | Just preference pairs |
| Performance | SOTA | Comparable or better |
| Recommended | Research | Production |

**DPO Loss Function**:
```python
# DPO directly optimizes on preference pairs
loss = -log(sigmoid(β * (log_prob_chosen - log_prob_rejected)))
```

### DeepSeek-R1 Training Pipeline (Latest SOTA)

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Cold Start SFT                                        │
│  - Small high-quality reasoning examples                        │
│  - Teaches format and clarity                                   │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Pure RL (GRPO)                                        │
│  - No SFT, just RL from scratch                                 │
│  - Model learns reasoning by doing                              │
│  - Verifiable rewards (math correctness)                        │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Rejection Sampling + Refinement                       │
│  - Filter bad outputs                                           │
│  - Human preference alignment                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Emerging Techniques to Watch

1. **Process Reward Models (PRMs)** - Reward each reasoning step, not just final answer
2. **Synthetic Data Generation** - Use strong models to generate training data
3. **Multi-Turn RL** - Optimize entire conversations, not single responses
4. **Tool-Augmented RL** - Models learn to use calculators, code interpreters
5. **Constitutional AI** - Models critique and improve their own outputs

---

## 📖 References

### Papers
- [InstructGPT](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022
- [DPO](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [LoRA](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [QLoRA](https://arxiv.org/abs/2305.14314) - Dettmers et al., 2023
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) - DeepSeek, 2025

### Libraries & Tools
- [TRL Library](https://github.com/huggingface/trl) - RLHF training
- [Alignment Handbook](https://github.com/huggingface/alignment-handbook) - HuggingFace
- [Open-R1](https://github.com/huggingface/open-r1) - Reasoning reproduction
- [verl](https://github.com/volcengine/verl) - Agentic RL framework

### Datasets
- [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) - Preference pairs
- [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) - 66K preferences
- [Orca DPO Pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) - GPT-4 vs Llama

---

## 📊 Hardware Used

All experiments run on **CPU** (Apple M-series / Intel). No GPU required!

| Notebook | Runtime | Memory |
|----------|---------|--------|
| 01-03 | 11-86 min | ~8GB RAM |
| 06 | 15 min | ~8GB RAM |
| 07 | 18 min | ~10GB RAM |

---

## 📝 License

MIT License - Use freely for learning and research!

---

**Made with 🧠 for understanding LLM training from scratch!**