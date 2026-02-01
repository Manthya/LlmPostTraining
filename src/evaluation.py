"""
Evaluation Utilities for LLM Post-Training
==========================================
Comprehensive evaluation framework to compare model responses across training stages.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from test_queries import (
    TEST_QUERIES,
    TestQuery,
    get_all_queries,
    get_queries_by_category,
    get_evaluation_prompt,
)


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""
    query_id: str
    category: str
    query: str
    expected_behavior: str
    model_response: str
    stage: int
    model_name: str
    generation_time: float
    prompt_tokens: int
    response_tokens: int
    timestamp: str


@dataclass
class EvaluationReport:
    """Complete evaluation report for a model."""
    model_name: str
    stage: int
    timestamp: str
    total_queries: int
    results: List[EvaluationResult]
    generation_config: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "total_queries": self.total_queries,
            "generation_config": self.generation_config,
            "results": [asdict(r) for r in self.results],
        }


class ModelEvaluator:
    """Evaluator for comparing model responses across stages."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_name: str,
        stage: int,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.stage = stage
        self.device = device
        
        self.model.eval()
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Tuple[str, float, int, int]:
        """Generate response and return (response, time, prompt_tokens, response_tokens)."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generation_time = time.time() - start_time
        
        # Decode full output
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            response = full_response
        
        response_tokens = outputs[0].shape[0] - prompt_tokens
        
        return response, generation_time, prompt_tokens, response_tokens
    
    def evaluate_query(
        self,
        query: TestQuery,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> EvaluationResult:
        """Evaluate a single query."""
        
        # Get stage-appropriate prompt
        prompt = get_evaluation_prompt(query, self.stage)
        
        # Generate response
        response, gen_time, prompt_tokens, response_tokens = self.generate_response(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        return EvaluationResult(
            query_id=query.id,
            category=query.category,
            query=query.query,
            expected_behavior=query.expected_behavior,
            model_response=response,
            stage=self.stage,
            model_name=self.model_name,
            generation_time=gen_time,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            timestamp=datetime.now().isoformat(),
        )
    
    def evaluate_all(
        self,
        queries: Optional[List[TestQuery]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> EvaluationReport:
        """Evaluate all queries and return report."""
        
        if queries is None:
            queries = get_all_queries()
        
        results = []
        
        for i, query in enumerate(queries):
            if verbose:
                print(f"Evaluating [{i+1}/{len(queries)}]: {query.id}")
            
            result = self.evaluate_query(
                query=query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results.append(result)
        
        report = EvaluationReport(
            model_name=self.model_name,
            stage=self.stage,
            timestamp=datetime.now().isoformat(),
            total_queries=len(results),
            results=results,
            generation_config={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        
        return report
    
    def evaluate_variants(
        self,
        query: TestQuery,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> List[EvaluationResult]:
        """Evaluate a query and all its paraphrase variants."""
        
        results = []
        
        # Original query
        results.append(self.evaluate_query(query, max_new_tokens, temperature))
        
        # Variants
        for i, variant in enumerate(query.variants):
            variant_query = TestQuery(
                id=f"{query.id}_variant_{i+1}",
                category=query.category,
                query=variant,
                expected_behavior=query.expected_behavior,
                stage1_expectation=query.stage1_expectation,
                stage2_expectation=query.stage2_expectation,
                stage3_expectation=query.stage3_expectation,
                variants=[],
                difficulty=query.difficulty,
            )
            results.append(self.evaluate_query(variant_query, max_new_tokens, temperature))
        
        return results


def save_evaluation_report(report: EvaluationReport, output_path: str):
    """Save evaluation report to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    
    print(f"Report saved to {output_path}")


def load_evaluation_report(path: str) -> Dict:
    """Load evaluation report from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_stages(
    base_report: EvaluationReport,
    stage1_report: Optional[EvaluationReport] = None,
    stage2_report: Optional[EvaluationReport] = None,
    stage3_report: Optional[EvaluationReport] = None,
) -> Dict[str, Any]:
    """Compare evaluation results across stages."""
    
    reports = {
        "base": base_report,
        "stage1": stage1_report,
        "stage2": stage2_report,
        "stage3": stage3_report,
    }
    
    # Filter out None reports
    reports = {k: v for k, v in reports.items() if v is not None}
    
    comparison = {
        "stages_compared": list(reports.keys()),
        "by_query": {},
        "by_category": {},
    }
    
    # Compare by query
    for query_id in [r.query_id for r in base_report.results]:
        comparison["by_query"][query_id] = {}
        
        for stage_name, report in reports.items():
            for result in report.results:
                if result.query_id == query_id:
                    comparison["by_query"][query_id][stage_name] = {
                        "response": result.model_response,
                        "generation_time": result.generation_time,
                    }
    
    return comparison


def print_comparison_table(
    query_id: str,
    base_response: str,
    stage1_response: Optional[str] = None,
    stage2_response: Optional[str] = None,
    stage3_response: Optional[str] = None,
):
    """Print a formatted comparison table for a single query."""
    
    print(f"\n{'='*80}")
    print(f"Query ID: {query_id}")
    print(f"{'='*80}")
    
    responses = [
        ("Base Model", base_response),
        ("Stage 1 (SFT)", stage1_response),
        ("Stage 2 (Instruction)", stage2_response),
        ("Stage 3 (LoRA)", stage3_response),
    ]
    
    for name, response in responses:
        if response is not None:
            print(f"\n{name}:")
            print(f"{'-'*40}")
            print(response[:500] + "..." if len(response) > 500 else response)
    
    print(f"\n{'='*80}")


def evaluate_instruction_robustness(
    evaluator: ModelEvaluator,
    query: TestQuery,
) -> Dict[str, Any]:
    """
    Evaluate how robust the model is to instruction paraphrasing.
    Key metric for Stage 2 success.
    """
    
    results = evaluator.evaluate_variants(query)
    
    responses = [r.model_response for r in results]
    
    # Calculate simple consistency metric (can be enhanced)
    # Here we just check response length variance as a proxy
    lengths = [len(r) for r in responses]
    avg_length = sum(lengths) / len(lengths)
    length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    
    return {
        "query_id": query.id,
        "num_variants": len(query.variants) + 1,
        "responses": responses,
        "avg_response_length": avg_length,
        "length_variance": length_variance,
        "results": [asdict(r) for r in results],
    }


if __name__ == "__main__":
    print("Evaluation utilities loaded.")
    print(f"Total test queries available: {len(TEST_QUERIES)}")
