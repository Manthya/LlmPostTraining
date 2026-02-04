"""
SFT Trainer for Reasoning Distillation
Following DeepSeek-R1-Distill methodology

LEARNINGS APPLIED FROM EXPERIMENTS:
1. KL Regularization to prevent catastrophic forgetting
2. Lower learning rate with cosine schedule
3. Gradient clipping for stability
4. Early stopping to prevent overfitting
5. Single consistent template (handled in data_preprocessing)

This module implements pure SFT training to distill reasoning capabilities
from the OpenThoughts dataset into a non-reasoning base model.
"""

import os
import logging
import copy
from typing import Optional, Dict, Any
from dataclasses import asdict

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
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


class KLRegularizedTrainer(Trainer):
    """
    Custom Trainer with KL Regularization
    
    LEARNED FROM INSTRUCTGPT EXPERIMENTS:
    Adding KL penalty prevents catastrophic forgetting by keeping
    the model close to the original pretrained weights.
    
    Loss = SFT_Loss + β * KL(current_model || reference_model)
    """
    
    def __init__(
        self,
        reference_model,
        kl_coef: float = 0.1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.reference_model = reference_model
        self.kl_coef = kl_coef
        logger.info(f"KLRegularizedTrainer initialized with kl_coef={kl_coef}")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with KL regularization
        
        Total Loss = CE_Loss + β * KL_Divergence
        
        This prevents the model from drifting too far from pretrained weights,
        which we learned causes catastrophic forgetting.
        """
        # Get model outputs
        outputs = model(**inputs)
        ce_loss = outputs.loss
        
        # Compute KL divergence if reference model exists
        if self.reference_model is not None and self.kl_coef > 0:
            with torch.no_grad():
                ref_outputs = self.reference_model(**inputs)
                ref_logits = ref_outputs.logits
            
            # Get current model logits
            current_logits = outputs.logits
            
            # Compute KL divergence (current || reference)
            # Using log_softmax for numerical stability
            current_log_probs = F.log_softmax(current_logits, dim=-1)
            ref_probs = F.softmax(ref_logits, dim=-1)
            
            # KL(P || Q) = sum(P * log(P/Q)) = sum(P * log(P)) - sum(P * log(Q))
            # We compute KL(current || ref)
            kl_div = F.kl_div(
                current_log_probs,
                ref_probs,
                reduction='batchmean',
                log_target=False
            )
            
            # Total loss with KL penalty
            total_loss = ce_loss + self.kl_coef * kl_div
            
            # Log KL divergence periodically
            if self.state.global_step % 100 == 0:
                logger.info(f"Step {self.state.global_step}: CE={ce_loss:.4f}, KL={kl_div:.4f}, Total={total_loss:.4f}")
        else:
            total_loss = ce_loss
        
        return (total_loss, outputs) if return_outputs else total_loss


class ReasoningDistillationTrainer:
    """
    Trainer for distilling reasoning capabilities using SFT
    Following DeepSeek-R1-Distill paper approach
    
    IMPROVEMENTS FROM EXPERIMENTS:
    - KL regularization to prevent catastrophic forgetting
    - Early stopping to prevent overfitting
    - Gradient clipping for stability
    - Reference model for KL computation
    """
    
    def __init__(self, config: DistillationConfig):
        """
        Initialize the trainer
        
        Args:
            config: Distillation configuration
        """
        self.config = config
        self.model = None
        self.reference_model = None  # For KL regularization
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
        Also loads reference model if KL regularization is enabled
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
            "low_cpu_mem_usage": True,  # MEMORY: Reduce CPU memory during loading
        }
        
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            **model_kwargs
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.config, 'gradient_checkpointing') and self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (memory optimization)")
        
        # Clear CUDA cache after loading
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Load reference model for KL regularization (LEARNED FROM INSTRUCTGPT)
        if self.config.use_kl_regularization and self.device == "cuda":
            logger.info("Loading reference model for KL regularization...")
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                **model_kwargs
            )
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
            logger.info("Reference model loaded and frozen")
        
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
        
        # Disable multiprocessing completely when CUDA is active
        # num_proc=None means single-process (no forking)
        # This avoids "Cannot re-initialize CUDA in forked subprocess" error
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=None,  # Disable multiprocessing for CUDA compatibility
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
        Setup the Hugging Face Trainer with improvements from experiments
        
        IMPROVEMENTS:
        - Early stopping callback
        - Gradient clipping
        - Better metric tracking
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        logger.info("Setting up trainer with optimized settings...")
        
        # Tokenize datasets
        train_dataset = self.prepare_dataset(train_dataset)
        if eval_dataset is not None:
            eval_dataset = self.prepare_dataset(eval_dataset)
        
        # Training arguments (OPTIMIZED FROM EXPERIMENTS)
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
            max_grad_norm=self.config.max_grad_norm,  # CRITICAL: Gradient clipping
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
            # MEMORY OPTIMIZATIONS
            gradient_checkpointing=getattr(self.config, 'gradient_checkpointing', True),
            optim="adamw_torch_fused" if self.device == "cuda" else "adamw_torch",  # Faster optimizer
            # Additional stability settings
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Callbacks
        callbacks = []
        if eval_dataset is not None:
            # Early stopping to prevent overfitting (LEARNED FROM EXPERIMENTS)
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold
            )
            callbacks.append(early_stopping)
            logger.info(f"Early stopping enabled: patience={self.config.early_stopping_patience}")
        
        # Use custom trainer with KL regularization if enabled
        if self.config.use_kl_regularization and self.reference_model is not None:
            logger.info(f"Using KL-regularized trainer with β={self.config.kl_coef}")
            self.trainer = KLRegularizedTrainer(
                model=self.model,
                reference_model=self.reference_model,
                kl_coef=self.config.kl_coef,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
            )
        else:
            # Standard trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
            )
        
        logger.info("Trainer setup complete")
    
    def train(self) -> Dict[str, Any]:
        """
        Run the training with enhanced logging
        
        IMPROVEMENTS FROM EXPERIMENTS:
        - Detailed logging of training progress
        - Early stopping support
        - Memory tracking
        
        Returns:
            Training metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer first.")
        
        logger.info("="*60)
        logger.info("STARTING TRAINING")
        logger.info("="*60)
        logger.info(f"  Model: {self.config.base_model_name}")
        logger.info(f"  Learning Rate: {self.config.learning_rate}")
        logger.info(f"  Epochs: {self.config.num_train_epochs}")
        logger.info(f"  Batch Size: {self.config.per_device_train_batch_size}")
        logger.info(f"  Gradient Accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  KL Regularization: {self.config.use_kl_regularization}")
        if self.config.use_kl_regularization:
            logger.info(f"  KL Coefficient: {self.config.kl_coef}")
        logger.info(f"  LoRA: {self.config.use_lora}")
        logger.info(f"  Gradient Checkpointing: {getattr(self.config, 'gradient_checkpointing', False)}")
        logger.info("="*60)
        
        # Track memory before training
        if self.device == "cuda":
            torch.cuda.empty_cache()  # Clear cache before training
            torch.cuda.reset_peak_memory_stats()
            logger.info(f"GPU Memory (start): {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        train_result = self.trainer.train()
        
        # Log final metrics
        metrics = train_result.metrics
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETED")
        logger.info("="*60)
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        # Log peak memory usage
        if self.device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"  Peak GPU Memory: {peak_memory:.2f} GB")
        
        logger.info("="*60)
        
        return metrics
    
    def save_model(self, output_path: Optional[str] = None) -> None:
        """
        Save the trained model with training configuration for reproducibility
        
        IMPROVEMENTS FROM EXPERIMENTS:
        - Save training configuration for reproducibility
        - Save training hyperparameters used
        - Track KL regularization settings if used
        
        Args:
            output_path: Optional custom output path
        """
        import json
        
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
        
        # Save training configuration for reproducibility
        training_config = {
            "base_model_name": self.config.base_model_name,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "warmup_ratio": self.config.warmup_ratio,
            "use_lora": self.config.use_lora,
            "use_kl_regularization": self.config.use_kl_regularization,
            "kl_coef": self.config.kl_coef if self.config.use_kl_regularization else None,
            "max_grad_norm": self.config.max_grad_norm,
            "early_stopping_patience": self.config.early_stopping_patience,
            "early_stopping_threshold": self.config.early_stopping_threshold,
            "seed": self.config.seed,
            "model_max_length": self.config.model_max_length,
        }
        
        config_path = os.path.join(output_path, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(training_config, f, indent=2)
        logger.info(f"Training configuration saved to {config_path}")
        
        logger.info("Model saved successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3
    ) -> str:
        """
        Generate a response for a given prompt with repetition controls
        
        IMPROVEMENTS FROM EXPERIMENTS:
        - Added repetition_penalty to prevent degenerate outputs
        - Added no_repeat_ngram_size to block repeated n-grams
        - Uses single consistent template (matching training)
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            do_sample: Whether to use sampling
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            no_repeat_ngram_size: Block repeating n-grams of this size
        
        Returns:
            Generated text
        """
        # Format prompt - USE SINGLE CONSISTENT TEMPLATE (matching training!)
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
        
        # Generate with repetition controls
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
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
