#!/usr/bin/env python3
"""
Benchmark Evaluation Script for Reasoning Distillation
=======================================================

Evaluates base model vs SFT model on standard reasoning benchmarks.
Computes Pass@1 accuracy and generates comparison report.

Supports two inference backends:
- HuggingFace Transformers (default)
- vLLM (3-5x faster, use --use_vllm flag)

Benchmarks:
- GSM8K: Grade school math word problems (1,319 test examples)
- MATH: Competition-level math (5,000 test examples)
- ARC-Challenge: Science reasoning (1,172 test examples)

Usage:
    # Standard HuggingFace inference
    python run_benchmark_evaluation.py \
        --base_model Qwen/Qwen2.5-0.5B \
        --sft_model ./outputs/distillation/run_XXXXXX/final_model \
        --benchmarks gsm8k
    
    # Fast vLLM inference (3-5x faster)
    python run_benchmark_evaluation.py \
        --base_model Qwen/Qwen2.5-0.5B \
        --sft_model ./outputs/distillation/run_XXXXXX/final_model \
        --benchmarks gsm8k \
        --use_vllm
"""

import os
import sys
import json
import re
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Optional vLLM import
VLLM_AVAILABLE = False
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration following DeepSeek-R1 methodology"""
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.1


# ============================================================================
# ANSWER EXTRACTION
# ============================================================================

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numerical answer from GSM8K response"""
    # Look for #### pattern (standard GSM8K format)
    match = re.search(r'####\s*([-\d,.\s]+)', text)
    if match:
        return match.group(1).replace(',', '').replace(' ', '').strip()
    
    # Look for "answer is" pattern
    match = re.search(r'(?:answer|result|solution)\s*(?:is|=|:)\s*([-\d,.\s]+)', text.lower())
    if match:
        return match.group(1).replace(',', '').replace(' ', '').strip()
    
    # Look for boxed answer
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    
    # After </think> tag, find last number
    think_end = text.find('</think>')
    if think_end != -1:
        after_think = text[think_end:]
        numbers = re.findall(r'[-+]?\d*\.?\d+', after_think)
        if numbers:
            return numbers[-1]
    
    # Fallback: last number in text
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1]
    
    return None


def extract_math_answer(text: str) -> Optional[str]:
    """Extract answer from MATH dataset response"""
    # Look for boxed answer
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    
    # After </think> tag
    think_end = text.find('</think>')
    if think_end != -1:
        after_think = text[think_end + 8:].strip()
        first_line = after_think.split('\n')[0].strip()
        if first_line:
            return first_line
    
    # Look for "final answer" pattern
    match = re.search(r'(?:final answer|answer)\s*(?:is|=|:)\s*([^\n.]+)', text.lower())
    if match:
        return match.group(1).strip()
    
    return None


def extract_arc_answer(text: str) -> Optional[str]:
    """Extract answer choice (A, B, C, D) from ARC response"""
    # After </think> tag
    think_end = text.find('</think>')
    if think_end != -1:
        after_think = text[think_end + 8:].strip()
        match = re.search(r'^([A-Da-d])\b', after_think)
        if match:
            return match.group(1).upper()
    
    # Look for explicit answer pattern
    match = re.search(r'(?:answer|choice)\s*(?:is|=|:)\s*\(?([A-Da-d])\)?', text)
    if match:
        return match.group(1).upper()
    
    # Look for "The answer is A" pattern
    match = re.search(r'(?:the\s+)?answer\s+is\s+\(?([A-Da-d])\)?', text.lower())
    if match:
        return match.group(1).upper()
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    answer = re.sub(r'^[\$\\]', '', answer)
    answer = re.sub(r'[\$\\]$', '', answer)
    answer = answer.replace(',', '')
    answer = ' '.join(answer.split())
    return answer


def check_answer(predicted: Optional[str], ground_truth: str, benchmark: str) -> bool:
    """Check if predicted answer matches ground truth"""
    if predicted is None:
        return False
    
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    if pred_norm == gt_norm:
        return True
    
    # Numerical comparison for math benchmarks
    if benchmark in ['gsm8k', 'math']:
        try:
            pred_float = float(pred_norm.replace(',', ''))
            gt_float = float(gt_norm.replace(',', ''))
            return abs(pred_float - gt_float) < 1e-6
        except (ValueError, TypeError):
            pass
    
    return False


# ============================================================================
# BENCHMARK DEFINITIONS
# ============================================================================

BENCHMARKS = {
    'gsm8k': {
        'dataset': 'openai/gsm8k',
        'subset': 'main',
        'split': 'test',
        'question_field': 'question',
        'answer_field': 'answer',
        'extractor': extract_gsm8k_answer,
        'description': 'Grade School Math (1,319 test examples)'
    },
    'math': {
        'dataset': 'lighteval/MATH',
        'subset': 'all',
        'split': 'test',
        'question_field': 'problem',
        'answer_field': 'solution',
        'extractor': extract_math_answer,
        'description': 'Competition Math (5,000 test examples)'
    },
    'arc_challenge': {
        'dataset': 'allenai/ai2_arc',
        'subset': 'ARC-Challenge',
        'split': 'test',
        'question_field': 'question',
        'answer_field': 'answerKey',
        'choices_field': 'choices',
        'extractor': extract_arc_answer,
        'description': 'AI2 Reasoning Challenge (1,172 test examples)'
    }
}


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(
    model_path: str,
    base_model_name: str = None,
    device: str = "cuda"
) -> Tuple[Any, Any]:
    """Load model and tokenizer, handling LoRA adapters"""
    
    logger.info(f"Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "low_cpu_mem_usage": True
    }
    
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    
    # Check if it's a LoRA adapter
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path) and base_model_name:
        logger.info("Loading as LoRA adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    logger.info("Model loaded successfully")
    
    return model, tokenizer


# ============================================================================
# vLLM MODEL LOADING AND INFERENCE
# ============================================================================

def load_vllm_model(
    model_path: str,
    base_model_name: str = None,
    gpu_memory_utilization: float = 0.9
) -> Tuple[Any, Optional[str]]:
    """
    Load model using vLLM for fast inference
    
    Returns:
        Tuple of (LLM instance, lora_path if applicable)
    """
    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM not installed. Install with: pip install vllm")
    
    logger.info(f"Loading model with vLLM: {model_path}")
    
    # Check if it's a LoRA adapter
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path) and base_model_name is not None
    
    if is_lora:
        # For LoRA, load base model with LoRA support enabled
        logger.info(f"Loading base model {base_model_name} with LoRA adapter support")
        llm = LLM(
            model=base_model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=64,
            dtype="float16"
        )
        return llm, model_path
    else:
        # Load as regular model
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="float16"
        )
        return llm, None


def unload_vllm_model(llm):
    """Unload vLLM model from memory"""
    del llm
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    logger.info("vLLM model unloaded from memory")


def generate_vllm_batch(
    llm,
    prompts: List[str],
    config: EvalConfig,
    lora_path: Optional[str] = None
) -> List[Tuple[str, float]]:
    """
    Generate responses for a batch of prompts using vLLM
    
    vLLM is optimized for batch inference - much faster than sequential
    """
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
        repetition_penalty=config.repetition_penalty
    )
    
    start_time = time.time()
    
    if lora_path:
        # Use LoRA adapter
        lora_request = LoRARequest("sft_adapter", 1, lora_path)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)
    
    total_time = time.time() - start_time
    avg_time = total_time / len(prompts)
    
    results = []
    for output in outputs:
        response = output.outputs[0].text
        results.append((response, avg_time))
    
    return results


def evaluate_model_vllm(
    llm,
    examples: List[Dict],
    model_name: str,
    config: EvalConfig,
    lora_path: Optional[str] = None,
    batch_size: int = 32
) -> Dict:
    """Evaluate model on benchmark using vLLM with batching"""
    
    benchmark = examples[0]['benchmark']
    extractor = BENCHMARKS[benchmark]['extractor']
    
    # Prepare all prompts
    prompts = []
    for example in examples:
        prompt = format_prompt(
            example['question'],
            benchmark,
            example.get('choices')
        )
        prompts.append(prompt)
    
    # Process in batches
    all_responses = []
    total_time = 0.0
    
    logger.info(f"Processing {len(prompts)} prompts in batches of {batch_size}")
    
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Evaluating {model_name}"):
        batch_prompts = prompts[i:i+batch_size]
        batch_results = generate_vllm_batch(llm, batch_prompts, config, lora_path)
        all_responses.extend(batch_results)
        total_time += sum(t for _, t in batch_results) * len(batch_results)
    
    # Evaluate results
    correct = 0
    reasoning_count = 0
    results = []
    
    for example, (response, gen_time) in zip(examples, all_responses):
        predicted = extractor(response)
        is_correct = check_answer(predicted, example['answer'], benchmark)
        has_reasoning = '<think>' in response.lower()
        
        if is_correct:
            correct += 1
        if has_reasoning:
            reasoning_count += 1
        
        results.append({
            'question': example['question'][:100],
            'ground_truth': example['answer'],
            'predicted': predicted,
            'is_correct': is_correct,
            'has_reasoning': has_reasoning
        })
    
    total = len(examples)
    return {
        'model_name': model_name,
        'benchmark': benchmark,
        'total': total,
        'correct': correct,
        'pass_at_1': correct / total * 100,
        'pct_with_reasoning': reasoning_count / total * 100,
        'avg_time': total_time / total if total > 0 else 0,
        'total_time': total_time,
        'detailed_results': results
    }


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def format_prompt(question: str, benchmark: str, choices: Dict = None) -> str:
    """Format prompt with system instruction for reasoning"""
    
    if benchmark == 'arc_challenge' and choices:
        choice_text = "\n".join([
            f"{label}) {text}" 
            for label, text in zip(choices['label'], choices['text'])
        ])
        question_with_choices = f"{question}\n\nChoices:\n{choice_text}"
        
        return f"""<|im_start|>system
You are a helpful assistant that thinks step by step before answering. When solving problems, first think through your reasoning inside <think> tags, then provide your final answer as just the letter (A, B, C, or D).<|im_end|>
<|im_start|>user
{question_with_choices}<|im_end|>
<|im_start|>assistant
"""
    else:
        return f"""<|im_start|>system
You are a helpful assistant that thinks step by step before answering. When solving problems, first think through your reasoning inside <think> tags, then provide your final numerical answer.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""


def generate_response(
    model,
    tokenizer,
    prompt: str,
    config: EvalConfig,
    device: str = "cuda"
) -> Tuple[str, float]:
    """Generate response from model"""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=config.repetition_penalty
        )
    
    generation_time = time.time() - start_time
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response, generation_time


def load_benchmark_data(benchmark: str, max_samples: int = None) -> List[Dict]:
    """Load benchmark dataset"""
    
    config = BENCHMARKS[benchmark]
    logger.info(f"Loading {benchmark}: {config['description']}")
    
    dataset = load_dataset(
        config['dataset'],
        config.get('subset'),
        split=config['split'],
        trust_remote_code=True
    )
    
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    examples = []
    for item in dataset:
        example = {
            'question': item[config['question_field']],
            'answer': item[config['answer_field']],
            'benchmark': benchmark
        }
        
        # Extract GSM8K answer
        if benchmark == 'gsm8k':
            match = re.search(r'####\s*([-\d,.\s]+)', example['answer'])
            if match:
                example['answer'] = match.group(1).replace(',', '').replace(' ', '').strip()
        
        # Handle ARC choices
        if 'choices_field' in config:
            example['choices'] = item[config['choices_field']]
        
        examples.append(example)
    
    return examples


def evaluate_model_on_benchmark(
    model,
    tokenizer,
    examples: List[Dict],
    model_name: str,
    config: EvalConfig,
    device: str = "cuda"
) -> Dict:
    """Evaluate a single model on benchmark examples"""
    
    benchmark = examples[0]['benchmark']
    extractor = BENCHMARKS[benchmark]['extractor']
    
    correct = 0
    total = len(examples)
    total_time = 0.0
    reasoning_count = 0
    
    results = []
    
    for example in tqdm(examples, desc=f"Evaluating {model_name}"):
        prompt = format_prompt(
            example['question'],
            benchmark,
            example.get('choices')
        )
        
        response, gen_time = generate_response(model, tokenizer, prompt, config, device)
        total_time += gen_time
        
        predicted = extractor(response)
        is_correct = check_answer(predicted, example['answer'], benchmark)
        has_reasoning = '<think>' in response.lower()
        
        if is_correct:
            correct += 1
        if has_reasoning:
            reasoning_count += 1
        
        results.append({
            'question': example['question'][:100],
            'ground_truth': example['answer'],
            'predicted': predicted,
            'is_correct': is_correct,
            'has_reasoning': has_reasoning
        })
    
    return {
        'model_name': model_name,
        'benchmark': benchmark,
        'total': total,
        'correct': correct,
        'pass_at_1': correct / total * 100,
        'pct_with_reasoning': reasoning_count / total * 100,
        'avg_time': total_time / total,
        'total_time': total_time,
        'detailed_results': results
    }


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def unload_model(model, device: str = "cuda"):
    """Unload model from memory"""
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    logger.info("Model unloaded from memory")


def run_evaluation(
    base_model_name: str,
    sft_model_path: str,
    benchmarks: List[str],
    max_samples: int = None,
    output_dir: str = "./outputs/benchmarks",
    use_vllm: bool = False,
    vllm_batch_size: int = 32
):
    """Run full evaluation pipeline - loads models one at a time for efficiency"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = EvalConfig()
    
    # Check vLLM availability
    if use_vllm and not VLLM_AVAILABLE:
        print("⚠️  vLLM not installed. Install with: pip install vllm")
        print("   Falling back to HuggingFace inference...")
        use_vllm = False
    
    print("\n" + "="*70)
    print("🔬 BENCHMARK EVALUATION PIPELINE")
    print("   Following DeepSeek-R1 Methodology")
    print("="*70)
    print(f"\n📦 Base Model: {base_model_name}")
    print(f"📦 SFT Model:  {sft_model_path}")
    print(f"📊 Benchmarks: {', '.join(benchmarks)}")
    print(f"🔢 Max Samples: {'All' if max_samples is None else max_samples}")
    print(f"🖥️  Device: {device}")
    print(f"🚀 Backend: {'vLLM (fast)' if use_vllm else 'HuggingFace'}")
    print(f"🌡️  Temperature: {config.temperature}, Top-p: {config.top_p}")
    print("="*70 + "\n")
    
    # Store all results
    all_results = {
        'config': {
            'base_model': base_model_name,
            'sft_model': sft_model_path,
            'benchmarks': benchmarks,
            'max_samples': max_samples,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'backend': 'vllm' if use_vllm else 'huggingface',
            'timestamp': datetime.now().isoformat()
        },
        'benchmarks': {}
    }
    
    # Evaluate on each benchmark
    for benchmark in benchmarks:
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARK: {benchmark.upper()}")
        print(f"   {BENCHMARKS[benchmark]['description']}")
        print("="*70 + "\n")
        
        # Load data first (before loading any model)
        examples = load_benchmark_data(benchmark, max_samples)
        
        if use_vllm:
            # ============================================================
            # vLLM EVALUATION (FAST)
            # ============================================================
            
            # Evaluate Base Model
            print(f"\n📥 Loading Base Model with vLLM...")
            base_llm, _ = load_vllm_model(base_model_name)
            
            print(f"\n🔄 Evaluating Base Model on {benchmark} (vLLM batch inference)...")
            base_results = evaluate_model_vllm(
                base_llm, examples, base_model_name, config,
                lora_path=None, batch_size=vllm_batch_size
            )
            
            print(f"\n🗑️  Unloading Base Model...")
            unload_vllm_model(base_llm)
            
            # Evaluate SFT Model
            print(f"\n📥 Loading SFT Model with vLLM...")
            sft_llm, lora_path = load_vllm_model(
                sft_model_path, 
                base_model_name=base_model_name
            )
            
            print(f"\n🔄 Evaluating SFT Model on {benchmark} (vLLM batch inference)...")
            sft_results = evaluate_model_vllm(
                sft_llm, examples, sft_model_path, config,
                lora_path=lora_path, batch_size=vllm_batch_size
            )
            
            print(f"\n🗑️  Unloading SFT Model...")
            unload_vllm_model(sft_llm)
            
        else:
            # ============================================================
            # HUGGINGFACE EVALUATION (STANDARD)
            # ============================================================
            
            # Evaluate Base Model
            print(f"\n📥 Loading Base Model...")
            base_model, base_tokenizer = load_model_and_tokenizer(
                base_model_name, device=device
            )
            
            print(f"\n🔄 Evaluating Base Model on {benchmark}...")
            base_results = evaluate_model_on_benchmark(
                base_model, base_tokenizer, examples,
                base_model_name, config, device
            )
            
            # Unload base model to free GPU memory
            print(f"\n🗑️  Unloading Base Model...")
            unload_model(base_model, device)
            del base_tokenizer
            
            # Evaluate SFT Model
            print(f"\n📥 Loading SFT Model...")
            sft_model, sft_tokenizer = load_model_and_tokenizer(
                sft_model_path, 
                base_model_name=base_model_name,
                device=device
            )
            
            print(f"\n🔄 Evaluating SFT Model on {benchmark}...")
            sft_results = evaluate_model_on_benchmark(
                sft_model, sft_tokenizer, examples,
                sft_model_path, config, device
            )
            
            # Unload SFT model to free GPU memory
            print(f"\n🗑️  Unloading SFT Model...")
            unload_model(sft_model, device)
            del sft_tokenizer
        
        # Store results (common for both backends)
        all_results['benchmarks'][benchmark] = {
            'base': {k: v for k, v in base_results.items() if k != 'detailed_results'},
            'sft': {k: v for k, v in sft_results.items() if k != 'detailed_results'},
            'detailed_base': base_results['detailed_results'],
            'detailed_sft': sft_results['detailed_results']
        }
        
        # Print intermediate results
        print(f"\n📈 {benchmark.upper()} Results:")
        print(f"   Base Model Pass@1: {base_results['pass_at_1']:.2f}%")
        print(f"   SFT Model Pass@1:  {sft_results['pass_at_1']:.2f}%")
        print(f"   Improvement: {sft_results['pass_at_1'] - base_results['pass_at_1']:+.2f}%")
    
    # ========================================================================
    # FINAL COMPARISON REPORT
    # ========================================================================
    
    print("\n" + "="*70)
    print("📊 FINAL EVALUATION COMPARISON REPORT")
    print("="*70)
    
    print(f"\n{'Benchmark':<20} {'Base Pass@1':>15} {'SFT Pass@1':>15} {'Δ':>10}")
    print("-"*60)
    
    total_base_correct = 0
    total_sft_correct = 0
    total_examples = 0
    
    for benchmark in benchmarks:
        base_res = all_results['benchmarks'][benchmark]['base']
        sft_res = all_results['benchmarks'][benchmark]['sft']
        
        delta = sft_res['pass_at_1'] - base_res['pass_at_1']
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.2f}%"
        
        print(f"{benchmark:<20} {base_res['pass_at_1']:>14.2f}% {sft_res['pass_at_1']:>14.2f}% {delta_str:>10}")
        
        total_base_correct += base_res['correct']
        total_sft_correct += sft_res['correct']
        total_examples += base_res['total']
    
    print("-"*60)
    
    # Overall accuracy
    overall_base = total_base_correct / total_examples * 100
    overall_sft = total_sft_correct / total_examples * 100
    overall_delta = overall_sft - overall_base
    
    print(f"{'OVERALL':<20} {overall_base:>14.2f}% {overall_sft:>14.2f}% {'+' if overall_delta >= 0 else ''}{overall_delta:.2f}%")
    
    print("\n" + "-"*60)
    print("REASONING ANALYSIS")
    print("-"*60)
    
    print(f"\n{'Benchmark':<20} {'Base % Reasoning':>18} {'SFT % Reasoning':>18}")
    print("-"*60)
    
    for benchmark in benchmarks:
        base_res = all_results['benchmarks'][benchmark]['base']
        sft_res = all_results['benchmarks'][benchmark]['sft']
        
        print(f"{benchmark:<20} {base_res['pct_with_reasoning']:>17.1f}% {sft_res['pct_with_reasoning']:>17.1f}%")
    
    print("\n" + "-"*60)
    print("GENERATION TIME")
    print("-"*60)
    
    print(f"\n{'Benchmark':<20} {'Base Avg (s)':>15} {'SFT Avg (s)':>15}")
    print("-"*60)
    
    for benchmark in benchmarks:
        base_res = all_results['benchmarks'][benchmark]['base']
        sft_res = all_results['benchmarks'][benchmark]['sft']
        
        print(f"{benchmark:<20} {base_res['avg_time']:>15.2f} {sft_res['avg_time']:>15.2f}")
    
    print("\n" + "="*70)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary (without detailed results)
    summary_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.json")
    summary = {
        'config': all_results['config'],
        'results': {
            bench: {
                'base': all_results['benchmarks'][bench]['base'],
                'sft': all_results['benchmarks'][bench]['sft']
            }
            for bench in benchmarks
        },
        'overall': {
            'base_pass_at_1': overall_base,
            'sft_pass_at_1': overall_sft,
            'improvement': overall_delta,
            'total_examples': total_examples
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Summary saved to: {summary_path}")
    
    # Save detailed results
    detailed_path = os.path.join(output_dir, f"benchmark_detailed_{timestamp}.json")
    with open(detailed_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Detailed results saved to: {detailed_path}")
    
    return all_results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation for reasoning distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard HuggingFace inference (slower)
  python run_benchmark_evaluation.py \\
      --base_model Qwen/Qwen2.5-0.5B \\
      --sft_model ./outputs/distillation/run_XXXXXX/final_model \\
      --benchmarks gsm8k \\
      --max_samples 100

  # Fast vLLM inference (3-5x faster, recommended)
  python run_benchmark_evaluation.py \\
      --base_model Qwen/Qwen2.5-0.5B \\
      --sft_model ./outputs/distillation/run_XXXXXX/final_model \\
      --benchmarks gsm8k \\
      --use_vllm

  # Full evaluation on all benchmarks with vLLM
  python run_benchmark_evaluation.py \\
      --base_model Qwen/Qwen2.5-0.5B \\
      --sft_model ./outputs/distillation/run_XXXXXX/final_model \\
      --benchmarks gsm8k math arc_challenge \\
      --use_vllm
        """
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name/path (e.g., Qwen/Qwen2.5-0.5B)"
    )
    
    parser.add_argument(
        "--sft_model",
        type=str,
        required=True,
        help="Path to SFT-trained model"
    )
    
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gsm8k"],
        choices=["gsm8k", "math", "arc_challenge"],
        help="Benchmarks to evaluate on"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per benchmark (default: all)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/benchmarks",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for fast inference (3-5x faster). Requires: pip install vllm"
    )
    
    parser.add_argument(
        "--vllm_batch_size",
        type=int,
        default=32,
        help="Batch size for vLLM inference (default: 32)"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        base_model_name=args.base_model,
        sft_model_path=args.sft_model,
        benchmarks=args.benchmarks,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        use_vllm=args.use_vllm,
        vllm_batch_size=args.vllm_batch_size
    )


if __name__ == "__main__":
    main()
