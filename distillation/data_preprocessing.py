"""
Data Preprocessing for Reasoning Distillation
Following DeepSeek-R1-Distill paper methodology

The OpenThoughts-114k dataset contains:
- problem: The reasoning problem/question
- solution: The step-by-step reasoning solution with <think> tags
- source: Dataset source (e.g., NuminaMath, AIME, etc.)
"""

import logging
from typing import Dict, List, Optional, Tuple
from datasets import Dataset, load_dataset
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Chat template for reasoning distillation
# Following DeepSeek-R1 format with <think> tags for chain-of-thought
REASONING_CHAT_TEMPLATE = """<|im_start|>system
You are a helpful assistant that thinks step by step before answering. When solving problems, first think through your reasoning inside <think> tags, then provide your final answer.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
{solution}<|im_end|>"""


def load_openthoughts_dataset(
    dataset_name: str = "open-thoughts/OpenThoughts-114k",
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = False
) -> Dataset:
    """
    Load the OpenThoughts-114k dataset
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to load
        max_samples: Maximum number of samples to load (None for all)
        streaming: Whether to use streaming mode
    
    Returns:
        Loaded dataset
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming
        )
        
        if max_samples is not None and not streaming:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"Selected {len(dataset)} samples")
        
        return dataset
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def clean_solution(solution: str) -> str:
    """
    Clean and normalize the solution text
    Following DeepSeek-R1-Distill preprocessing
    
    Args:
        solution: Raw solution text
    
    Returns:
        Cleaned solution text
    """
    # Remove excessive whitespace
    solution = re.sub(r'\n{3,}', '\n\n', solution)
    solution = re.sub(r' {2,}', ' ', solution)
    
    # Normalize think tags
    solution = re.sub(r'<\s*think\s*>', '<think>', solution, flags=re.IGNORECASE)
    solution = re.sub(r'<\s*/\s*think\s*>', '</think>', solution, flags=re.IGNORECASE)
    
    # Ensure proper think tag structure
    if '<think>' not in solution.lower():
        # If no think tags, wrap the reasoning part
        # Look for common patterns that indicate reasoning vs answer
        answer_patterns = [
            r'(The answer is.*)',
            r'(Therefore,.*)',
            r'(Thus,.*)',
            r'(So the answer.*)',
            r'(\\boxed\{.*\})',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, solution, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = solution[:match.start()].strip()
                answer = match.group(0).strip()
                solution = f"<think>\n{reasoning}\n</think>\n\n{answer}"
                break
        else:
            # If no clear answer pattern, wrap entire solution in think tags
            solution = f"<think>\n{solution}\n</think>"
    
    return solution.strip()


def format_for_sft(
    example: Dict,
    tokenizer=None,
    max_length: int = 2048
) -> Dict:
    """
    Format a single example for SFT training
    Following DeepSeek-R1-Distill format
    
    Args:
        example: Single dataset example with 'problem' and 'solution' keys
        tokenizer: Optional tokenizer for length validation
        max_length: Maximum sequence length
    
    Returns:
        Formatted example with 'text' key
    """
    problem = example.get('problem', example.get('question', ''))
    solution = example.get('solution', example.get('answer', ''))
    
    # Clean the solution
    solution = clean_solution(solution)
    
    # Format using chat template
    text = REASONING_CHAT_TEMPLATE.format(
        problem=problem.strip(),
        solution=solution
    )
    
    return {'text': text}


def preprocess_dataset(
    dataset: Dataset,
    tokenizer=None,
    max_length: int = 2048,
    num_proc: int = 4
) -> Dataset:
    """
    Preprocess the entire dataset for SFT
    
    Args:
        dataset: Raw dataset
        tokenizer: Tokenizer for processing
        max_length: Maximum sequence length
        num_proc: Number of processes for parallel processing
    
    Returns:
        Preprocessed dataset
    """
    logger.info("Preprocessing dataset...")
    
    def process_example(example):
        return format_for_sft(example, tokenizer, max_length)
    
    processed_dataset = dataset.map(
        process_example,
        num_proc=num_proc,
        desc="Formatting examples"
    )
    
    # Filter out examples that are too long if tokenizer is provided
    if tokenizer is not None:
        def filter_by_length(example):
            tokens = tokenizer(example['text'], truncation=False)
            return len(tokens['input_ids']) <= max_length
        
        original_size = len(processed_dataset)
        processed_dataset = processed_dataset.filter(
            filter_by_length,
            num_proc=num_proc,
            desc="Filtering by length"
        )
        filtered_size = len(processed_dataset)
        logger.info(f"Filtered {original_size - filtered_size} examples exceeding max length")
    
    return processed_dataset


def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into training and validation sets
    
    Args:
        dataset: Full dataset
        val_ratio: Ratio of validation samples
        seed: Random seed
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"Splitting dataset with {val_ratio:.1%} validation ratio")
    
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    
    logger.info(f"Train size: {len(split['train'])}, Val size: {len(split['test'])}")
    
    return split['train'], split['test']


def get_sample_prompts() -> List[Dict[str, str]]:
    """
    Get sample reasoning prompts for evaluation
    
    Returns:
        List of sample prompts with expected reasoning patterns
    """
    return [
        {
            "problem": "If a train travels at 60 mph for 2 hours and then at 80 mph for 1.5 hours, what is the total distance traveled?",
            "category": "math",
            "expected_pattern": "step-by-step calculation"
        },
        {
            "problem": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
            "category": "logic",
            "expected_pattern": "careful reading comprehension"
        },
        {
            "problem": "Solve for x: 3x + 7 = 22",
            "category": "algebra",
            "expected_pattern": "algebraic manipulation"
        },
        {
            "problem": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "category": "logic",
            "expected_pattern": "rate analysis"
        },
        {
            "problem": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "category": "logic",
            "expected_pattern": "careful algebraic setup"
        }
    ]


if __name__ == "__main__":
    # Test the preprocessing
    print("Testing data preprocessing...")
    
    # Load a small sample
    dataset = load_openthoughts_dataset(max_samples=10)
    print(f"Loaded {len(dataset)} samples")
    
    # Process
    processed = preprocess_dataset(dataset)
    print(f"Processed {len(processed)} samples")
    
    # Show a sample
    print("\n" + "="*50)
    print("Sample processed text:")
    print("="*50)
    print(processed[0]['text'][:500] + "...")
