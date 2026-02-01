"""
Training Utilities for LLM Post-Training
==========================================
Common training functions and helpers for all stages.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from datasets import Dataset


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_directories(config: Dict[str, Any]):
    """Create output directories if they don't exist."""
    output_config = config.get("output", {})
    
    for key, path in output_config.items():
        if path and isinstance(path, str):
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_base_model(
    model_name: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> AutoModelForCausalLM:
    """Load base model for full fine-tuning (Stage 1 & 2)."""
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    
    logger.info(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_map.get(torch_dtype, torch.bfloat16),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    
    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")
    
    return model


def load_quantized_model(
    model_name: str,
    quantization_config: Dict[str, Any],
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> AutoModelForCausalLM:
    """Load quantized model for QLoRA (Stage 3)."""
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_config.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(
            torch, quantization_config.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
        bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", True),
    )
    
    logger.info(f"Loading quantized model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    logger.info(f"Quantized model loaded. Parameters: {model.num_parameters():,}")
    
    return model


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
) -> AutoTokenizer:
    """Load tokenizer for the model."""
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def apply_lora(
    model: AutoModelForCausalLM,
    lora_config: Dict[str, Any],
) -> PeftModel:
    """Apply LoRA adapters to model (Stage 3)."""
    
    config = LoraConfig(
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("lora_alpha", 128),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
    )
    
    logger.info(f"Applying LoRA with rank {config.r}")
    
    model = get_peft_model(model, config)
    
    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA applied. Trainable: {trainable_params:,} / {all_params:,} "
        f"({100 * trainable_params / all_params:.2f}%)"
    )
    
    return model


# ============================================================================
# Training Functions
# ============================================================================

def create_training_arguments(
    config: Dict[str, Any],
    output_dir: str,
    run_name: Optional[str] = None,
) -> TrainingArguments:
    """Create TrainingArguments from config."""
    
    training_config = config.get("training", {})
    
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        
        # Batch size
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        
        # Training
        num_train_epochs=training_config.get("num_train_epochs", 3),
        learning_rate=training_config.get("learning_rate", 2e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        
        # Logging
        logging_steps=training_config.get("logging_steps", 10),
        eval_strategy="steps" if training_config.get("eval_steps") else "no",
        eval_steps=training_config.get("eval_steps", 100),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        
        # Mixed precision
        bf16=training_config.get("bf16", True),
        fp16=training_config.get("fp16", False),
        
        # Optimization
        optim=training_config.get("optim", "adamw_torch"),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        
        # Best model
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        
        # Reporting
        report_to=training_config.get("report_to", "none"),
    )
    
    return args


def train_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    training_args: TrainingArguments,
) -> Trainer:
    """Train the model using HuggingFace Trainer."""
    
    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    return trainer


def save_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    is_peft: bool = False,
):
    """Save model and tokenizer."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if is_peft:
        # Save only LoRA adapters
        model.save_pretrained(output_dir)
        logger.info(f"LoRA adapters saved to {output_dir}")
    else:
        # Save full model
        model.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
    
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer saved to {output_dir}")


def merge_lora_and_save(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    output_dir: str,
):
    """Merge LoRA adapters into base model and save."""
    
    logger.info("Merging LoRA adapters with base model...")
    
    merged_model = model.merge_and_unload()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Merged model saved to {output_dir}")


# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int = 512,
) -> float:
    """Compute perplexity on a list of texts."""
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate a response from the model."""
    
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response


# ============================================================================
# Memory Utilities
# ============================================================================

def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logger.info(
                f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
            )
    else:
        logger.info("CUDA not available")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")


if __name__ == "__main__":
    # Test loading
    print("Training utilities loaded successfully!")
    print_gpu_memory()
