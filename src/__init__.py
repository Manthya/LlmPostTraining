# Make src a package
from .data_utils import (
    load_config,
    load_jsonl,
    save_jsonl,
    format_stage1_sample,
    format_stage2_sample,
    create_stage1_dataset,
    create_stage2_dataset,
    create_stage3_dataset,
    load_alpaca_dataset,
    create_train_eval_split,
)

from .training_utils import (
    load_base_model,
    load_quantized_model,
    load_tokenizer,
    apply_lora,
    create_training_arguments,
    train_model,
    save_model,
    merge_lora_and_save,
    generate_response,
    compute_perplexity,
    print_gpu_memory,
    clear_gpu_memory,
)

from .test_queries import (
    TEST_QUERIES,
    TestQuery,
    get_all_queries,
    get_queries_by_category,
    get_queries_by_difficulty,
    get_evaluation_prompt,
    export_queries_to_json,
)

from .evaluation import (
    ModelEvaluator,
    EvaluationResult,
    EvaluationReport,
    save_evaluation_report,
    load_evaluation_report,
    compare_stages,
    evaluate_instruction_robustness,
)

__all__ = [
    # Data utilities
    "load_config",
    "load_jsonl",
    "save_jsonl",
    "format_stage1_sample",
    "format_stage2_sample",
    "create_stage1_dataset",
    "create_stage2_dataset",
    "create_stage3_dataset",
    "load_alpaca_dataset",
    "create_train_eval_split",
    
    # Training utilities
    "load_base_model",
    "load_quantized_model",
    "load_tokenizer",
    "apply_lora",
    "create_training_arguments",
    "train_model",
    "save_model",
    "merge_lora_and_save",
    "generate_response",
    "compute_perplexity",
    "print_gpu_memory",
    "clear_gpu_memory",
    
    # Test queries
    "TEST_QUERIES",
    "TestQuery",
    "get_all_queries",
    "get_queries_by_category",
    "get_queries_by_difficulty",
    "get_evaluation_prompt",
    "export_queries_to_json",
    
    # Evaluation
    "ModelEvaluator",
    "EvaluationResult",
    "EvaluationReport",
    "save_evaluation_report",
    "load_evaluation_report",
    "compare_stages",
    "evaluate_instruction_robustness",
]
