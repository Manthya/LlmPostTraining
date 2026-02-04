"""
Data Utilities for LLM Post-Training
=====================================
Handles data loading, formatting, and preprocessing for all training stages.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field

import yaml
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    format: str  # "prompt_output" or "instruction_response"
    max_length: int = 512
    train_split: float = 0.9
    seed: int = 42
    datasets: List[Dict[str, str]] = field(default_factory=list)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================================
# Stage 1: Prompt → Output Format
# ============================================================================

def format_stage1_sample(sample: Dict, prompt_template: Optional[str] = None) -> str:
    """
    Format a sample for Stage 1 (Normal SFT).
    Simple prompt → output format, no chat structure.
    """
    prompt = sample.get("prompt", sample.get("question", ""))
    output = sample.get("output", sample.get("answer", sample.get("response", "")))
    
    if prompt_template:
        return prompt_template.format(prompt=prompt, output=output)
    
    # Default minimal format
    return f"{prompt}\n\nAnswer: {output}"


def create_stage1_dataset(
    data: List[Dict],
    tokenizer,
    max_length: int = 512,
) -> Dataset:
    """
    Create dataset for Stage 1 training.
    """
    formatted_texts = []
    
    for sample in data:
        text = format_stage1_sample(sample)
        formatted_texts.append({"text": text})
    
    dataset = Dataset.from_list(formatted_texts)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    return tokenized_dataset


# ============================================================================
# Stage 2: Instruction → Response Format
# ============================================================================

INSTRUCTION_TEMPLATES = [
    # Alpaca-style
    """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}""",
    
    # Simple format
    """Instruction: {instruction}
{input}

Response: {response}""",
    
    # ChatML-style
    """<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{instruction}
{input}
<|im_end|>
<|im_start|>assistant
{response}
<|im_end|>""",
]


SYSTEM_MESSAGES = [
    "You are a helpful assistant.",
    "You are an AI assistant that follows instructions carefully.",
    "You are a knowledgeable and helpful AI.",
    "You are an assistant designed to help users with their tasks.",
]


def format_stage2_sample(
    sample: Dict,
    template_idx: Optional[int] = None,
    randomize: bool = True,
) -> str:
    """
    Format a sample for Stage 2 (Instruction Tuning).
    Uses instruction → input (optional) → response format.
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    response = sample.get("response", sample.get("output", ""))
    
    if randomize and template_idx is None:
        template_idx = random.randint(0, len(INSTRUCTION_TEMPLATES) - 1)
    elif template_idx is None:
        template_idx = 0
    
    template = INSTRUCTION_TEMPLATES[template_idx]
    
    return template.format(
        instruction=instruction,
        input=input_text if input_text else "",
        response=response,
    )


def create_stage2_dataset(
    data: List[Dict],
    tokenizer,
    max_length: int = 1024,
    randomize_templates: bool = True,
) -> Dataset:
    """
    Create dataset for Stage 2 training with template randomization.
    """
    formatted_texts = []
    
    for sample in data:
        text = format_stage2_sample(sample, randomize=randomize_templates)
        formatted_texts.append({"text": text})
    
    dataset = Dataset.from_list(formatted_texts)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    return tokenized_dataset


# ============================================================================
# Stage 3: Same as Stage 2 (LoRA/QLoRA uses same data format)
# ============================================================================

def create_stage3_dataset(
    data: List[Dict],
    tokenizer,
    max_length: int = 1024,
    randomize_templates: bool = True,
) -> Dataset:
    """
    Create dataset for Stage 3 (LoRA/QLoRA).
    Uses same format as Stage 2 for comparison.
    """
    return create_stage2_dataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        randomize_templates=randomize_templates,
    )


# ============================================================================
# Dataset Loading Utilities
# ============================================================================

def load_alpaca_dataset(subset_size: Optional[int] = None) -> List[Dict]:
    """Load cleaned Alpaca dataset from HuggingFace."""
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    
    data = []
    for item in dataset:
        data.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "response": item["output"],
        })
    
    if subset_size:
        random.shuffle(data)
        data = data[:subset_size]
    
    return data


def load_openassistant_dataset(subset_size: Optional[int] = None) -> List[Dict]:
    """Load OpenAssistant dataset for instruction tuning."""
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    
    # Filter for English, assistant responses
    data = []
    for item in dataset:
        if item["lang"] == "en" and item["role"] == "assistant":
            data.append({
                "instruction": "Respond helpfully to the user.",
                "input": item.get("parent_text", ""),
                "response": item["text"],
            })
    
    if subset_size:
        random.shuffle(data)
        data = data[:subset_size]
    
    return data


def create_train_eval_split(
    data: List[Dict],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple:
    """Split data into training and evaluation sets."""
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    split_idx = int(len(data_copy) * train_ratio)
    train_data = data_copy[:split_idx]
    eval_data = data_copy[split_idx:]
    
    return train_data, eval_data


# ============================================================================
# Sample Data Generation (for testing)
# ============================================================================

def generate_sample_stage1_data(n_samples: int = 100) -> List[Dict]:
    """Generate sample data for Stage 1 testing."""
    samples = []
    
    # QA samples
    qa_pairs = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is 2 + 2?", "2 + 2 equals 4."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
        ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
    ]
    
    # Task completion samples
    task_pairs = [
        ("Translate 'Hello' to Spanish:", "Hola"),
        ("Summarize in one word: happy, joyful, delighted", "Happy"),
        ("Calculate: 15 * 3", "45"),
        ("Complete the sequence: 1, 2, 4, 8, ...", "16"),
        ("Convert 100 Celsius to Fahrenheit:", "212°F"),
    ]
    
    all_pairs = qa_pairs + task_pairs
    
    for i in range(n_samples):
        pair = all_pairs[i % len(all_pairs)]
        samples.append({
            "prompt": pair[0],
            "output": pair[1],
        })
    
    return samples


def generate_sample_stage2_data(n_samples: int = 100) -> List[Dict]:
    """Generate sample data for Stage 2 testing."""
    samples = []
    
    instructions = [
        {
            "instruction": "Translate the following text to French.",
            "input": "Hello, how are you?",
            "response": "Bonjour, comment allez-vous?",
        },
        {
            "instruction": "Summarize the following text in one sentence.",
            "input": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
            "response": "Machine learning is an AI subset that allows systems to learn from data and experience without explicit programming.",
        },
        {
            "instruction": "Write a haiku about spring.",
            "input": "",
            "response": "Cherry blossoms fall\nGentle breeze carries petals\nSpring awakens now",
        },
        {
            "instruction": "Explain the concept to a 5-year-old.",
            "input": "What is gravity?",
            "response": "Gravity is like an invisible hand that pulls everything down to the ground. That's why when you throw a ball up, it always comes back down!",
        },
        {
            "instruction": "Convert this sentence to passive voice.",
            "input": "The cat chased the mouse.",
            "response": "The mouse was chased by the cat.",
        },
    ]
    
    for i in range(n_samples):
        sample = instructions[i % len(instructions)].copy()
        samples.append(sample)
    
    return samples


if __name__ == "__main__":
    # Generate sample data for testing
    stage1_data = generate_sample_stage1_data(100)
    stage2_data = generate_sample_stage2_data(100)
    
    # Save sample data
    save_jsonl(stage1_data, "./data/stage1/sample_data.jsonl")
    save_jsonl(stage2_data, "./data/stage2/sample_data.jsonl")
    
    print("Sample data generated successfully!")
    print(f"Stage 1: {len(stage1_data)} samples")
    print(f"Stage 2: {len(stage2_data)} samples")
