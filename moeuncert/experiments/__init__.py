"""Utilities for experiment scripts."""

from .paths import (
    resolve_model_slug,
    resolve_dataset_slug,
    resolve_default_period,
    resolve_project_root,
    resolve_pretrained_models_dir,
    resolve_cache_dir,
    build_data_path,
    build_results_path,
    build_figures_path,
    load_tokenizer_for_data,
)
from .utils import (
    optimal_threshold,
    get_quantization_kwargs,
    read_and_collate_outputs,
    iter_batch_outputs,
    stratified_group_split,
    compute_metrics_at_threshold,
    create_token_labels,
    find_generation_boundaries,
    replace_inf_with_nan,
)
from .data_loading import (
    load_labeled_dataset,
    load_model_outputs,
    load_multi_year_data,
    stream_multi_year_data,
    stream_sampled_outputs,
)


__all__ = [
    "resolve_model_slug",
    "resolve_dataset_slug",
    "resolve_default_period",
    "resolve_project_root",
    "resolve_pretrained_models_dir",
    "resolve_cache_dir",
    "build_data_path",
    "build_results_path",
    "build_figures_path",
    "load_tokenizer_for_data",
    "optimal_threshold",
    "get_quantization_kwargs",
    "read_and_collate_outputs",
    "iter_batch_outputs",
    "stratified_group_split",
    "compute_metrics_at_threshold",
    "create_token_labels",
    "find_generation_boundaries",
    "load_labeled_dataset",
    "load_model_outputs",
    "load_multi_year_data",
    "stream_multi_year_data",
    "stream_sampled_outputs",
    "replace_inf_with_nan",
]
