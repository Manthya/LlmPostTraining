#!/usr/bin/env python3
"""
Reasoning Distillation Pipeline
================================

Complete pipeline for distilling reasoning capabilities from the OpenThoughts-114k
dataset into a non-reasoning base model (Qwen/Llama).

Following DeepSeek-R1-Distill paper methodology:
1. Load non-reasoning base model
2. Load and preprocess OpenThoughts-114k dataset
3. Run SFT training with chain-of-thought reasoning data
4. Evaluate and compare base vs SFT model on reasoning tasks

Usage:
    python run_distillation_pipeline.py --mode all
    python run_distillation_pipeline.py --mode train
    python run_distillation_pipeline.py --mode evaluate
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Optional

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DistillationConfig, EvaluationConfig
from data_preprocessing import (
    load_openthoughts_dataset,
    preprocess_dataset,
    create_train_val_split
)
from sft_trainer import ReasoningDistillationTrainer
from evaluation import ReasoningEvaluator, get_reasoning_test_prompts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('distillation_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║          Reasoning Distillation Pipeline v1.0                   ║
║          Following DeepSeek-R1-Distill Methodology              ║
╠══════════════════════════════════════════════════════════════════╣
║  Dataset: OpenThoughts-114k                                      ║
║  Method: Pure SFT with Chain-of-Thought Reasoning               ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_device():
    """Check and report available device"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✓ CUDA available: {gpu_name} ({gpu_memory:.2f} GB)")
    else:
        device = "cpu"
        logger.info("⚠ CUDA not available, using CPU (training will be slow)")
    
    return device


def run_training(
    config: DistillationConfig,
    resume_from: Optional[str] = None
) -> str:
    """
    Run the SFT training phase
    
    Args:
        config: Training configuration
        resume_from: Optional checkpoint to resume from
    
    Returns:
        Path to saved model
    """
    logger.info("="*60)
    logger.info("STAGE 1: SFT Training")
    logger.info("="*60)
    
    # Initialize trainer
    trainer = ReasoningDistillationTrainer(config)
    
    # Load model and tokenizer
    logger.info(f"Loading base model: {config.base_model_name}")
    trainer.load_model_and_tokenizer()
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset_name}")
    dataset = load_openthoughts_dataset(
        dataset_name=config.dataset_name,
        max_samples=config.max_samples
    )
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Preprocess
    logger.info("Preprocessing dataset...")
    processed_dataset = preprocess_dataset(
        dataset,
        tokenizer=trainer.tokenizer,
        max_length=config.model_max_length
    )
    logger.info(f"Preprocessed {len(processed_dataset)} samples")
    
    # Split
    train_dataset, val_dataset = create_train_val_split(
        processed_dataset,
        val_ratio=config.validation_split,
        seed=config.seed
    )
    
    # Setup trainer
    trainer.setup_trainer(train_dataset, val_dataset)
    
    # Train
    logger.info("Starting training...")
    metrics = trainer.train()
    
    # Save
    model_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(model_path)
    
    logger.info(f"Training completed! Model saved to: {model_path}")
    logger.info(f"Final metrics: {metrics}")
    
    return model_path


def run_evaluation(
    base_model_name: str,
    sft_model_path: Optional[str] = None,
    output_path: str = None,
    device: str = None
) -> dict:
    """
    Run evaluation comparing base and SFT models
    
    Args:
        base_model_name: Base model name/path
        sft_model_path: Path to SFT model
        output_path: Path to save results
        device: Device to use
    
    Returns:
        Evaluation results
    """
    logger.info("="*60)
    logger.info("STAGE 2: Evaluation")
    logger.info("="*60)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize evaluator
    evaluator = ReasoningEvaluator(
        base_model_name=base_model_name,
        sft_model_path=sft_model_path,
        device=device
    )
    
    # Load models
    logger.info("Loading base model for evaluation...")
    evaluator.load_base_model()
    
    if sft_model_path and os.path.exists(sft_model_path):
        logger.info("Loading SFT model for evaluation...")
        evaluator.load_sft_model()
    else:
        logger.warning("No SFT model path provided or path doesn't exist")
    
    # Get test prompts
    test_prompts = get_reasoning_test_prompts()
    logger.info(f"Evaluating on {len(test_prompts)} reasoning tasks")
    
    # Run evaluation
    results = evaluator.evaluate_batch(test_prompts)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    if "summary" in results:
        summary = results["summary"]
        
        if "base_model" in summary:
            print(f"\n📊 Base Model ({base_model_name}):")
            print(f"   - Avg Reasoning Quality Score: {summary['base_model']['avg_reasoning_quality']:.3f}")
            print(f"   - % with Reasoning: {summary['base_model']['pct_with_reasoning']:.1f}%")
            print(f"   - Avg Generation Time: {summary['base_model']['avg_generation_time']:.2f}s")
        
        if "sft_model" in summary:
            print(f"\n📊 SFT Model ({sft_model_path}):")
            print(f"   - Avg Reasoning Quality Score: {summary['sft_model']['avg_reasoning_quality']:.3f}")
            print(f"   - % with Reasoning: {summary['sft_model']['pct_with_reasoning']:.1f}%")
            print(f"   - Avg Generation Time: {summary['sft_model']['avg_generation_time']:.2f}s")
        
        if "improvement" in summary:
            print(f"\n📈 Improvement (SFT vs Base):")
            print(f"   - Reasoning Quality Delta: {summary['improvement']['reasoning_quality_delta']:+.3f}")
            print(f"   - Reasoning % Delta: {summary['improvement']['reasoning_pct_delta']:+.1f}%")
    
    # Save results
    if output_path:
        evaluator.save_results(results, output_path)
    
    return results


def run_full_pipeline(args):
    """Run the complete distillation pipeline"""
    print_banner()
    
    device = check_device()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"./outputs/distillation/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure training
    config = DistillationConfig(
        base_model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        device=device
    )
    
    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        # Convert dataclass to dict, handling non-serializable types
        config_dict = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v 
                      for k, v in config.__dict__.items()}
        json.dump(config_dict, f, indent=2)
    logger.info(f"Config saved to {config_path}")
    
    sft_model_path = None
    
    # Training phase
    if args.mode in ["all", "train"]:
        sft_model_path = run_training(config)
    
    # Evaluation phase
    if args.mode in ["all", "evaluate"]:
        if sft_model_path is None:
            sft_model_path = args.sft_model_path or os.path.join(output_dir, "final_model")
        
        eval_output = os.path.join(output_dir, "evaluation_results.json")
        
        results = run_evaluation(
            base_model_name=args.model,
            sft_model_path=sft_model_path if os.path.exists(sft_model_path) else None,
            output_path=eval_output,
            device=device
        )
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning Distillation Pipeline - DeepSeek-R1 Style",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with defaults
  python run_distillation_pipeline.py --mode all

  # Train only with custom settings
  python run_distillation_pipeline.py --mode train --model Qwen/Qwen2.5-0.5B --epochs 3 --max_samples 1000

  # Evaluate existing models
  python run_distillation_pipeline.py --mode evaluate --sft_model_path ./outputs/distillation/final_model

  # Quick test run
  python run_distillation_pipeline.py --mode all --max_samples 100 --epochs 1
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "train", "evaluate"],
        default="all",
        help="Pipeline mode: 'all' (train + evaluate), 'train' only, or 'evaluate' only"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base model name/path (default: Qwen/Qwen2.5-0.5B)"
    )
    
    parser.add_argument(
        "--sft_model_path",
        type=str,
        default=None,
        help="Path to existing SFT model (for evaluate mode)"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="open-thoughts/OpenThoughts-114k",
        help="Dataset name (default: open-thoughts/OpenThoughts-114k)"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum training samples (None for full dataset)"
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2)"
    )
    
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient training (default: True)"
    )
    
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated with timestamp)"
    )
    
    args = parser.parse_args()
    
    # Handle no_lora flag
    if args.no_lora:
        args.use_lora = False
    
    # Run pipeline
    run_full_pipeline(args)


if __name__ == "__main__":
    main()
