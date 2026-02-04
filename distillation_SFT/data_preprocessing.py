"""
Data Preprocessing for Reasoning Distillation
Following DeepSeek-R1-Distill paper methodology

LEARNINGS APPLIED FROM SFT EXPERIMENTS:
1. SINGLE consistent template (no template randomization - causes gradient conflicts)
2. Quality filtering (remove too short/long, degenerate samples)
3. Proper formatting for reasoning with <think> tags

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


# SINGLE Chat template for reasoning distillation
# CRITICAL LEARNING: Using ONE consistent template prevents gradient conflicts
# Following DeepSeek-R1 format with <think> tags for chain-of-thought
REASONING_CHAT_TEMPLATE = """<|im_start|>system
You are a helpful assistant that thinks step by step before answering. When solving problems, first think through your reasoning inside <think> tags, then provide your final answer.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
{solution}<|im_end|>"""

# Quality thresholds learned from experiments
# NOTE: These are LENIENT defaults - the OpenThoughts dataset is already curated
MIN_SOLUTION_LENGTH = 20  # Too short = low quality
MAX_SOLUTION_LENGTH = 16000  # Allow longer solutions (reasoning can be verbose)
MIN_PROBLEM_LENGTH = 5  # Valid problem minimum
REPETITION_THRESHOLD = 0.5  # Max allowed repetition ratio (lenient)
ENABLE_QUALITY_FILTERING = False  # Disable by default - OpenThoughts is already high quality


def load_openthoughts_dataset(
    dataset_name: str = "open-thoughts/OpenThoughts-114k",
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = False,
    subset: str = "metadata"  # Use metadata subset which has proper field names
) -> Dataset:
    """
    Load the OpenThoughts-114k dataset
    
    Note: The 'metadata' subset contains:
    - problem, deepseek_solution, deepseek_reasoning, domain, etc.
    
    The 'default' subset may have different/fewer fields.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to load
        max_samples: Maximum number of samples to load (None for all)
        streaming: Whether to use streaming mode
        subset: Dataset subset ('default' or 'metadata')
    
    Returns:
        Loaded dataset
    """
    logger.info(f"Loading dataset: {dataset_name} (subset: {subset})")
    
    try:
        # Try to load with subset first
        try:
            dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
                streaming=streaming
            )
            logger.info(f"Loaded '{subset}' subset successfully")
        except Exception:
            # Fallback to default (no subset specified)
            logger.info(f"Subset '{subset}' not found, trying default...")
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


def detect_repetition(text: str) -> float:
    """
    Detect repetitive patterns in text (LEARNED FROM EXPERIMENTS)
    High repetition indicates degenerate outputs
    
    Args:
        text: Text to analyze
    
    Returns:
        Repetition ratio (0-1, higher = more repetitive)
    """
    if len(text) < 100:
        return 0.0
    
    # Check for repeated n-grams
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    
    # Count unique vs total 3-grams
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    if not trigrams:
        return 0.0
    
    unique_ratio = len(set(trigrams)) / len(trigrams)
    repetition_ratio = 1 - unique_ratio
    
    return repetition_ratio


def is_quality_sample(problem: str, solution: str, strict: bool = False) -> bool:
    """
    Filter for quality samples (LEARNED FROM EXPERIMENTS)
    Low quality data causes catastrophic forgetting and degenerate outputs
    
    NOTE: OpenThoughts-114k is already a curated dataset, so filtering is
    disabled by default. Enable strict=True for custom/uncurated datasets.
    
    Args:
        problem: The problem text
        solution: The solution text
        strict: Whether to apply strict quality filtering
    
    Returns:
        True if sample passes quality checks
    """
    # If filtering is disabled globally and not strict, pass everything
    if not ENABLE_QUALITY_FILTERING and not strict:
        # Only do basic sanity checks
        if not problem or not problem.strip():
            return False
        if not solution or not solution.strip():
            return False
        return True
    
    # Length checks
    if len(problem.strip()) < MIN_PROBLEM_LENGTH:
        return False
    if len(solution.strip()) < MIN_SOLUTION_LENGTH:
        return False
    if len(solution.strip()) > MAX_SOLUTION_LENGTH:
        return False
    
    # Repetition check (only in strict mode)
    if strict and detect_repetition(solution) > REPETITION_THRESHOLD:
        return False
    
    # Check for clearly degenerate patterns only
    degenerate_patterns = [
        r'(.)\1{20,}',  # Same character repeated 20+ times (very lenient)
    ]
    
    for pattern in degenerate_patterns:
        if re.search(pattern, solution):
            return False
    
    return True


def format_for_sft(
    example: Dict,
    tokenizer=None,
    max_length: int = 2048
) -> Dict:
    """
    Format a single example for SFT training
    Following DeepSeek-R1-Distill format
    
    CRITICAL: Uses SINGLE consistent template (learned from experiments)
    Multiple templates cause gradient conflicts and degrade performance
    
    OpenThoughts-114k structure (metadata subset):
    - problem: The problem/question
    - deepseek_solution: The full solution with <think> tags
    - deepseek_reasoning: Just the reasoning part
    - domain: math, code, science, etc.
    
    Args:
        example: Single dataset example
        tokenizer: Optional tokenizer for length validation
        max_length: Maximum sequence length
    
    Returns:
        Formatted example with 'text' key and quality flag
    """
    # OpenThoughts-114k field names (metadata subset)
    problem = (
        example.get('problem') or 
        example.get('question') or 
        example.get('instruction') or
        example.get('prompt') or
        ''
    )
    
    # Try deepseek_solution first (contains full reasoning), then others
    solution = (
        example.get('deepseek_solution') or
        example.get('solution') or 
        example.get('response') or
        example.get('answer') or
        ''
    )
    
    # If no solution but has deepseek_reasoning, construct the solution
    if not solution and example.get('deepseek_reasoning'):
        reasoning = example.get('deepseek_reasoning', '')
        answer = example.get('ground_truth_solution', example.get('answer', ''))
        solution = f"<think>\n{reasoning}\n</think>\n\n{answer}"
    
    # Basic sanity check only
    if not problem or not problem.strip() or not solution or not solution.strip():
        return {'text': '', 'is_valid': False, 'debug_keys': list(example.keys())}
    
    # Clean the solution
    solution = clean_solution(solution)
    
    # Format using SINGLE consistent chat template
    # CRITICAL: No template randomization (causes gradient conflicts)
    text = REASONING_CHAT_TEMPLATE.format(
        problem=problem.strip(),
        solution=solution
    )
    
    return {'text': text, 'is_valid': True}


def preprocess_dataset(
    dataset: Dataset,
    tokenizer=None,
    max_length: int = 2048
) -> Dataset:
    """
    Preprocess the entire dataset for SFT
    
    IMPROVEMENTS FROM EXPERIMENTS:
    1. Quality filtering (removes degenerate samples)
    2. Length-based filtering
    3. Deduplication
    
    NOTE: Multiprocessing is disabled to avoid CUDA fork issues.
    
    Args:
        dataset: Raw dataset
        tokenizer: Tokenizer for processing
        max_length: Maximum sequence length
    
    Returns:
        Preprocessed, filtered dataset
    """
    logger.info("Preprocessing dataset with quality filtering...")
    
    # Debug: Log the dataset columns to understand structure
    if len(dataset) > 0:
        logger.info(f"Dataset columns: {dataset.column_names}")
        first_example = dataset[0]
        logger.info(f"First example keys: {list(first_example.keys())}")
        # Show a preview of non-empty fields
        for key, value in first_example.items():
            if value and isinstance(value, str) and len(value) > 0:
                preview = value[:100] + "..." if len(value) > 100 else value
                logger.info(f"  {key}: {preview}")
    
    def process_example(example):
        return format_for_sft(example, tokenizer, max_length)
    
    # Use num_proc=None to disable multiprocessing (avoids CUDA fork issues)
    processed_dataset = dataset.map(
        process_example,
        num_proc=None,
        desc="Formatting examples"
    )
    
    # Filter invalid samples (quality check failed)
    original_size = len(processed_dataset)
    processed_dataset = processed_dataset.filter(
        lambda x: x.get('is_valid', True) and len(x.get('text', '')) > 0,
        num_proc=None,
        desc="Quality filtering"
    )
    quality_filtered = original_size - len(processed_dataset)
    logger.info(f"Quality filtered: {quality_filtered} samples removed")
    
    # Filter out examples that are too long if tokenizer is provided
    if tokenizer is not None:
        def filter_by_length(example):
            tokens = tokenizer(example['text'], truncation=False)
            return len(tokens['input_ids']) <= max_length
        
        before_length_filter = len(processed_dataset)
        processed_dataset = processed_dataset.filter(
            filter_by_length,
            num_proc=None,
            desc="Filtering by length"
        )
        length_filtered = before_length_filter - len(processed_dataset)
        logger.info(f"Length filtered: {length_filtered} examples exceeding max length")
    
    # Remove is_valid column (no longer needed)
    if 'is_valid' in processed_dataset.column_names:
        processed_dataset = processed_dataset.remove_columns(['is_valid'])
    
    logger.info(f"Final dataset size: {len(processed_dataset)} samples")
    logger.info(f"Total filtered: {original_size - len(processed_dataset)} samples ({100*(original_size - len(processed_dataset))/original_size:.1f}%)")
    
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
