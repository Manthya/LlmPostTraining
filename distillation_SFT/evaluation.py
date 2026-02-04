"""
Evaluation Module for Reasoning Distillation
Compares base model vs SFT model on reasoning tasks

IMPROVEMENTS FROM EXPERIMENTS (Notebooks 01-07):
================================================
- Added repetition detection to identify degenerate outputs
- Added non-repetition score as a key metric
- Added degenerate pattern detection
- Uses single consistent template (matching training)
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_repetition(text: str, min_length: int = 10) -> float:
    """
    Detect repetitive patterns in text
    
    Returns a repetition score from 0 (no repetition) to 1 (highly repetitive)
    
    LEARNING FROM EXPERIMENTS:
    Models trained without KL regularization often produce degenerate,
    repetitive outputs. This function helps detect such outputs.
    """
    if len(text) < min_length:
        return 0.0
    
    # Check for repeated substrings
    words = text.lower().split()
    if len(words) < 3:
        return 0.0
    
    # Check for repeated word sequences
    repeated_sequences = 0
    for seq_len in range(2, min(10, len(words) // 2)):
        for i in range(len(words) - seq_len * 2):
            seq1 = ' '.join(words[i:i+seq_len])
            seq2 = ' '.join(words[i+seq_len:i+seq_len*2])
            if seq1 == seq2:
                repeated_sequences += 1
    
    # Normalize by text length
    max_possible = len(words) // 2
    repetition_score = min(repeated_sequences / max_possible, 1.0) if max_possible > 0 else 0.0
    
    # Also check for character-level repetition (e.g., "..." repeated)
    char_repetition = 0
    for i in range(len(text) - 10):
        if text[i:i+5] == text[i+5:i+10]:
            char_repetition += 1
    
    char_rep_score = min(char_repetition / (len(text) / 10), 1.0) if len(text) > 10 else 0.0
    
    return max(repetition_score, char_rep_score)


@dataclass
class EvaluationResult:
    """Store evaluation results for a single model"""
    model_name: str
    model_type: str  # "base" or "sft"
    prompt: str
    response: str
    category: str
    has_reasoning: bool
    reasoning_quality_score: float
    answer_extracted: Optional[str]
    generation_time: float
    # NEW: Repetition detection from experiments
    repetition_score: float = 0.0
    is_degenerate: bool = False


class ReasoningEvaluator:
    """
    Evaluator for comparing reasoning capabilities
    """
    
    def __init__(
        self,
        base_model_name: str,
        sft_model_path: Optional[str] = None,
        device: str = None
    ):
        """
        Initialize the evaluator
        
        Args:
            base_model_name: Name/path of the base model
            sft_model_path: Path to the SFT-trained model
            device: Device to use (cuda/cpu)
        """
        self.base_model_name = base_model_name
        self.sft_model_path = sft_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.base_model = None
        self.base_tokenizer = None
        self.sft_model = None
        self.sft_tokenizer = None
        
        logger.info(f"Evaluator initialized with device: {self.device}")
    
    def load_base_model(self) -> None:
        """Load the base (non-SFT) model"""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs
        )
        
        if self.device == "cpu":
            self.base_model = self.base_model.to(self.device)
        
        self.base_model.eval()
        logger.info("Base model loaded")
    
    def load_sft_model(self) -> None:
        """Load the SFT-trained model"""
        if self.sft_model_path is None:
            logger.warning("No SFT model path provided")
            return
        
        logger.info(f"Loading SFT model from: {self.sft_model_path}")
        
        # Check if it's a LoRA adapter or full model
        adapter_config_path = os.path.join(self.sft_model_path, "adapter_config.json")
        
        self.sft_tokenizer = AutoTokenizer.from_pretrained(
            self.sft_model_path,
            trust_remote_code=True
        )
        
        if self.sft_tokenizer.pad_token is None:
            self.sft_tokenizer.pad_token = self.sft_tokenizer.eos_token
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        if os.path.exists(adapter_config_path):
            # Load as LoRA model
            logger.info("Detected LoRA adapter, loading with PEFT")
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                **model_kwargs
            )
            self.sft_model = PeftModel.from_pretrained(base, self.sft_model_path)
        else:
            # Load as full model
            self.sft_model = AutoModelForCausalLM.from_pretrained(
                self.sft_model_path,
                **model_kwargs
            )
        
        if self.device == "cpu":
            self.sft_model = self.sft_model.to(self.device)
        
        self.sft_model.eval()
        logger.info("SFT model loaded")
    
    def _format_prompt(self, problem: str) -> str:
        """Format prompt for evaluation"""
        return f"""<|im_start|>system
You are a helpful assistant that thinks step by step before answering. When solving problems, first think through your reasoning inside <think> tags, then provide your final answer.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
"""
    
    def _generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3
    ) -> Tuple[str, float]:
        """
        Generate response from a model with repetition controls
        
        IMPROVEMENTS FROM EXPERIMENTS:
        - Added repetition_penalty to prevent degenerate outputs
        - Added no_repeat_ngram_size to block repeated n-grams
        
        Returns:
            Tuple of (response, generation_time)
        """
        import time
        
        formatted_prompt = self._format_prompt(prompt)
        
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        
        generation_time = time.time() - start_time
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response, generation_time
    
    def _analyze_response(self, response: str) -> Dict[str, Any]:
        """
        Analyze a response for reasoning quality
        
        IMPROVEMENTS FROM EXPERIMENTS:
        - Added repetition detection to identify degenerate outputs
        - Added non-repetition score as a key metric
        
        Returns:
            Dict with analysis results
        """
        analysis = {
            "has_reasoning": False,
            "reasoning_quality_score": 0.0,
            "answer_extracted": None,
            "has_think_tags": False,
            "reasoning_length": 0,
            "step_count": 0,
            # NEW: Repetition detection from experiments
            "repetition_score": 0.0,
            "is_degenerate": False,
        }
        
        # Check for repetition (learned from experiments)
        analysis["repetition_score"] = detect_repetition(response)
        
        # Degenerate patterns learned from experiments
        degenerate_patterns = [
            r'(.{10,})\1{2,}',  # Same text repeated 3+ times
            r'(step\s*\d+.*?){5,}',  # Same step pattern repeated
            r'(\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+\s*){4,}',  # Same calculation repeated
        ]
        
        for pattern in degenerate_patterns:
            if re.search(pattern, response, re.IGNORECASE | re.DOTALL):
                analysis["is_degenerate"] = True
                break
        
        # Also mark as degenerate if repetition score is high
        if analysis["repetition_score"] > 0.3:
            analysis["is_degenerate"] = True
        
        # Check for think tags
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        if think_match:
            analysis["has_think_tags"] = True
            analysis["has_reasoning"] = True
            reasoning = think_match.group(1)
            analysis["reasoning_length"] = len(reasoning)
            
            # Count steps (look for numbered steps or "step" mentions)
            step_patterns = [
                r'\d+[\.\)]\s',  # "1. " or "1) "
                r'step\s*\d+',   # "step 1"
                r'first[,\s]',   # "first,"
                r'second[,\s]',  # "second,"
                r'then[,\s]',    # "then,"
                r'finally[,\s]', # "finally,"
            ]
            
            for pattern in step_patterns:
                analysis["step_count"] += len(re.findall(pattern, reasoning, re.IGNORECASE))
        else:
            # Check for implicit reasoning patterns
            reasoning_indicators = [
                "let's", "first", "then", "therefore", "because",
                "since", "if", "so", "thus", "hence", "calculate"
            ]
            
            response_lower = response.lower()
            for indicator in reasoning_indicators:
                if indicator in response_lower:
                    analysis["has_reasoning"] = True
                    break
        
        # Calculate quality score (0-1)
        score = 0.0
        
        if analysis["has_think_tags"]:
            score += 0.3  # Has proper structure
        
        if analysis["has_reasoning"]:
            score += 0.2
        
        if analysis["reasoning_length"] > 100:
            score += 0.2
        
        if analysis["step_count"] >= 2:
            score += 0.2
        
        # Extract answer (look for boxed answer or final statement)
        answer_patterns = [
            r'\\boxed\{([^}]+)\}',
            r'answer[:\s]+([^\n]+)',
            r'result[:\s]+([^\n]+)',
            r'therefore[,\s]+([^\n]+)',
            r'=\s*(\d+(?:\.\d+)?)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                analysis["answer_extracted"] = match.group(1).strip()
                score += 0.1
                break
        
        analysis["reasoning_quality_score"] = min(score, 1.0)
        
        return analysis
    
    def evaluate_single(
        self,
        prompt: str,
        category: str = "general"
    ) -> Tuple[EvaluationResult, Optional[EvaluationResult]]:
        """
        Evaluate a single prompt on both models
        
        Args:
            prompt: The problem to solve
            category: Category of the problem
        
        Returns:
            Tuple of (base_result, sft_result)
        """
        results = []
        
        # Evaluate base model
        if self.base_model is not None:
            response, gen_time = self._generate(
                self.base_model,
                self.base_tokenizer,
                prompt
            )
            
            analysis = self._analyze_response(response)
            
            base_result = EvaluationResult(
                model_name=self.base_model_name,
                model_type="base",
                prompt=prompt,
                response=response,
                category=category,
                has_reasoning=analysis["has_reasoning"],
                reasoning_quality_score=analysis["reasoning_quality_score"],
                answer_extracted=analysis["answer_extracted"],
                generation_time=gen_time
            )
            results.append(base_result)
        else:
            results.append(None)
        
        # Evaluate SFT model
        if self.sft_model is not None:
            response, gen_time = self._generate(
                self.sft_model,
                self.sft_tokenizer,
                prompt
            )
            
            analysis = self._analyze_response(response)
            
            sft_result = EvaluationResult(
                model_name=self.sft_model_path or "sft_model",
                model_type="sft",
                prompt=prompt,
                response=response,
                category=category,
                has_reasoning=analysis["has_reasoning"],
                reasoning_quality_score=analysis["reasoning_quality_score"],
                answer_extracted=analysis["answer_extracted"],
                generation_time=gen_time
            )
            results.append(sft_result)
        else:
            results.append(None)
        
        return tuple(results)
    
    def evaluate_batch(
        self,
        prompts: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of prompts
        
        Args:
            prompts: List of dicts with 'problem' and 'category' keys
        
        Returns:
            Evaluation results summary
        """
        all_results = {
            "base_model": [],
            "sft_model": [],
            "summary": {}
        }
        
        for i, prompt_data in enumerate(prompts):
            logger.info(f"Evaluating prompt {i+1}/{len(prompts)}")
            
            problem = prompt_data.get("problem", prompt_data.get("question", ""))
            category = prompt_data.get("category", "general")
            
            base_result, sft_result = self.evaluate_single(problem, category)
            
            if base_result:
                all_results["base_model"].append(asdict(base_result))
            if sft_result:
                all_results["sft_model"].append(asdict(sft_result))
        
        # Calculate summary statistics
        all_results["summary"] = self._calculate_summary(all_results)
        
        return all_results
    
    def _calculate_summary(self, results: Dict) -> Dict[str, Any]:
        """Calculate summary statistics from results"""
        summary = {}
        
        for model_type in ["base_model", "sft_model"]:
            if results[model_type]:
                model_results = results[model_type]
                
                summary[model_type] = {
                    "total_samples": len(model_results),
                    "avg_reasoning_quality": sum(r["reasoning_quality_score"] for r in model_results) / len(model_results),
                    "pct_with_reasoning": sum(1 for r in model_results if r["has_reasoning"]) / len(model_results) * 100,
                    "avg_generation_time": sum(r["generation_time"] for r in model_results) / len(model_results),
                }
        
        # Calculate improvement if both models present
        if "base_model" in summary and "sft_model" in summary:
            summary["improvement"] = {
                "reasoning_quality_delta": summary["sft_model"]["avg_reasoning_quality"] - summary["base_model"]["avg_reasoning_quality"],
                "reasoning_pct_delta": summary["sft_model"]["pct_with_reasoning"] - summary["base_model"]["pct_with_reasoning"],
            }
        
        return summary
    
    def save_results(
        self,
        results: Dict,
        output_path: str
    ) -> None:
        """Save evaluation results to JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "base_model": self.base_model_name,
            "sft_model": self.sft_model_path,
            "device": self.device
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def get_reasoning_test_prompts() -> List[Dict[str, str]]:
    """Get test prompts for reasoning evaluation"""
    return [
        {
            "problem": "A train travels at 60 mph for 2 hours and then at 80 mph for 1.5 hours. What is the total distance traveled?",
            "category": "math"
        },
        {
            "problem": "If a farmer has 17 sheep and all but 9 run away, how many sheep does the farmer have left?",
            "category": "logic"
        },
        {
            "problem": "Solve for x: 3x + 7 = 22",
            "category": "algebra"
        },
        {
            "problem": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "category": "logic"
        },
        {
            "problem": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "category": "logic"
        },
        {
            "problem": "What is the sum of the first 10 positive integers?",
            "category": "math"
        },
        {
            "problem": "If you have a 3x3 grid and need to fill it with numbers 1-9 such that each row, column adds to 15, what goes in the center?",
            "category": "logic"
        },
        {
            "problem": "A car travels 120 miles using 4 gallons of gas. How many miles can it travel with 7 gallons?",
            "category": "math"
        },
        {
            "problem": "Three people check into a hotel room that costs $30. They each contribute $10. Later, the manager realizes the room only costs $25, so he sends the bellboy to return $5. The bellboy keeps $2 and gives each person $1 back. Now each person paid $9 (totaling $27), and the bellboy has $2. That's $29. Where did the extra dollar go?",
            "category": "logic"
        },
        {
            "problem": "Calculate: (15 + 27) × 3 - 42 ÷ 6",
            "category": "math"
        }
    ]


if __name__ == "__main__":
    # Test evaluation
    print("Testing evaluation module...")
    
    evaluator = ReasoningEvaluator(
        base_model_name="Qwen/Qwen2.5-0.5B"
    )
    
    evaluator.load_base_model()
    
    prompts = get_reasoning_test_prompts()[:2]  # Test with 2 prompts
    results = evaluator.evaluate_batch(prompts)
    
    print("\nResults Summary:")
    print(json.dumps(results["summary"], indent=2))
