"""
SFT Trainer for Reasoning Distillation
Following DeepSeek-R1-Distill methodology

This module implements pure SFT training to distill reasoning capabilities
from the OpenThoughts dataset into a non-reasoning base model.
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import asdict

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from config import DistillationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningDistillationTrainer:
    """
    Trainer for distilling reasoning capabilities using SFT
    Following DeepSeek-R1-Distill paper approach
    """
    
    def __init__(self, config: DistillationConfig):
        """
        Initialize the trainer
        
        Args:
            config: Distillation configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Set device
        self.device = config.device
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cuda":
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def load_model_and_tokenizer(self) -> None:
        """
        Load the base model and tokenizer
        """
        logger.info(f"Loading model: {self.config.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Model loading configuration
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.config.fp16 else torch.float32,
        }
        
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            **model_kwargs
        )
        
        # Apply LoRA if configured
        if self.config.use_lora:
            self._apply_lora()
        
        logger.info(f"Model loaded with {self._count_parameters()} parameters")
    
    def _apply_lora(self) -> None:
        """
        Apply LoRA adapters to the model
        """
        logger.info("Applying LoRA configuration...")
        
        # Prepare model for training
        if self.device == "cuda":
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params, all_params = self._count_trainable_params()
        logger.info(
            f"LoRA applied: {trainable_params:,} trainable parameters "
            f"({100 * trainable_params / all_params:.2f}% of {all_params:,})"
        )
    
    def _count_parameters(self) -> int:
        """Count total model parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def _count_trainable_params(self) -> tuple:
        """Count trainable and total parameters"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for training by tokenizing
        
        Args:
            dataset: Preprocessed dataset with 'text' field
        
        Returns:
            Tokenized dataset
        """
        logger.info("Tokenizing dataset...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.model_max_length,
                padding="max_length",
                return_tensors=None
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        return tokenized_dataset
    
    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> None:
        """
        Setup the Hugging Face Trainer
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        logger.info("Setting up trainer...")
        
        # Tokenize datasets
        train_dataset = self.prepare_dataset(train_dataset)
        if eval_dataset is not None:
            eval_dataset = self.prepare_dataset(eval_dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=self.config.eval_steps if eval_dataset is not None else None,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            seed=self.config.seed,
            report_to="none",  # Disable wandb/tensorboard by default
            load_best_model_at_end=True if eval_dataset is not None else False,
            metric_for_best_model="eval_loss" if eval_dataset is not None else None,
            greater_is_better=False,
            dataloader_pin_memory=True if self.device == "cuda" else False,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        logger.info("Trainer setup complete")
    
    def train(self) -> Dict[str, Any]:
        """
        Run the training
        
        Returns:
            Training metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer first.")
        
        logger.info("Starting training...")
        
        train_result = self.trainer.train()
        
        # Log metrics
        metrics = train_result.metrics
        logger.info(f"Training completed. Metrics: {metrics}")
        
        return metrics
    
    def save_model(self, output_path: Optional[str] = None) -> None:
        """
        Save the trained model
        
        Args:
            output_path: Optional custom output path
        """
        output_path = output_path or os.path.join(self.config.output_dir, "final_model")
        
        logger.info(f"Saving model to {output_path}")
        
        if self.config.use_lora:
            # Save LoRA adapters
            self.model.save_pretrained(output_path)
            
            # Also save merged model
            merged_path = os.path.join(self.config.output_dir, "merged_model")
            logger.info(f"Merging and saving full model to {merged_path}")
            
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(merged_path)
            self.tokenizer.save_pretrained(merged_path)
        else:
            self.model.save_pretrained(output_path)
        
        self.tokenizer.save_pretrained(output_path)
        
        logger.info("Model saved successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response for a given prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            do_sample: Whether to use sampling
        
        Returns:
            Generated text
        """
        # Format prompt
        formatted_prompt = f"""<|im_start|>system
You are a helpful assistant that thinks step by step before answering. When solving problems, first think through your reasoning inside <think> tags, then provide your final answer.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model_max_length
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text


def train_reasoning_distillation(config: DistillationConfig) -> None:
    """
    Main training function
    
    Args:
        config: Distillation configuration
    """
    from data_preprocessing import (
        load_openthoughts_dataset,
        preprocess_dataset,
        create_train_val_split
    )
    
    # Initialize trainer
    trainer = ReasoningDistillationTrainer(config)
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    # Load and preprocess dataset
    dataset = load_openthoughts_dataset(
        dataset_name=config.dataset_name,
        max_samples=config.max_samples
    )
    
    processed_dataset = preprocess_dataset(
        dataset,
        tokenizer=trainer.tokenizer,
        max_length=config.model_max_length
    )
    
    # Split into train/val
    train_dataset, val_dataset = create_train_val_split(
        processed_dataset,
        val_ratio=config.validation_split,
        seed=config.seed
    )
    
    # Setup and train
    trainer.setup_trainer(train_dataset, val_dataset)
    metrics = trainer.train()
    
    # Save model
    trainer.save_model()
    
    return trainer, metrics


if __name__ == "__main__":
    # Test training setup
    config = DistillationConfig(
        max_samples=100,  # Small sample for testing
        num_train_epochs=1,
        logging_steps=5,
        save_steps=50
    )
    
    trainer, metrics = train_reasoning_distillation(config)
    print(f"Training completed with metrics: {metrics}")
