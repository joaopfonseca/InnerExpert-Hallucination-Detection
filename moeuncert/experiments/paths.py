"""Path and slug resolution utilities for experiment scripts."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


def resolve_model_slug(model_name: str) -> str:
    """
    Convert model name to filesystem-safe slug.
    
    Parameters
    ----------
    model_name : str
        Model name or path (e.g., "allenai/OLMoE-1B-7B-0924-Instruct")
    
    Returns
    -------
    str
        Filesystem-safe slug (e.g., "allenai__OLMoE-1B-7B-0924-Instruct")
    
    Examples
    --------
    >>> resolve_model_slug("allenai/OLMoE-1B-7B-0924-Instruct")
    'allenai__OLMoE-1B-7B-0924-Instruct'
    """
    return model_name.replace("/", "__")


def resolve_default_period() -> Tuple[List[int], int]:
    """
    Resolve default dataset period to last month.
    
    Returns
    -------
    Tuple[List[int], int]
        (years, month) tuple representing the previous calendar month
    
    Examples
    --------
    >>> # If today is 2026-03-20
    >>> resolve_default_period()
    ([2026], 2)
    
    >>> # If today is 2026-01-15
    >>> resolve_default_period()
    ([2025], 12)
    """
    time_now = datetime.now()
    if time_now.month > 1:
        month = time_now.month - 1
        year = time_now.year
    else:
        month = 12
        year = time_now.year - 1
    return [year], month


def resolve_dataset_slug(
    years: Optional[List[int]] = None,
    month: Optional[int] = None,
) -> Tuple[List[int], Optional[int], str]:
    """
    Resolve dataset period and generate slug following naming conventions.
    
    Parameters
    ----------
    years : List[int], optional
        Year(s) for the dataset. If None, defaults to last month via
        resolve_default_period()
    month : int, optional
        Month (1-12). Only valid when exactly one year is provided.
        For multi-year datasets, month must be None.
    
    Returns
    -------
    Tuple[List[int], Optional[int], str]
        (years, month, dataset_slug) where:
        - years: Resolved list of years
        - month: Resolved month (or None for multi-year)
        - dataset_slug: String identifier following conventions:
            * Single month: "realtimeqa-YYYY-MM"
            * Multiple years: "realtimeqa-Y1-Y2-..."
    
    Raises
    ------
    ValueError
        If month is specified with multiple years
    
    Examples
    --------
    >>> resolve_dataset_slug([2026], 2)
    ([2026], 2, 'realtimeqa-2026-02')
    
    >>> resolve_dataset_slug([2024, 2025, 2026])
    ([2024, 2025, 2026], None, 'realtimeqa-2024-2025-2026')
    
    >>> resolve_dataset_slug()  # Uses current month
    ([2026], 2, 'realtimeqa-2026-02')  # if today is 2026-03-20
    """
    if years is None:
        years, month = resolve_default_period()
    else:
        years = list(years)
    
    if len(years) > 1 and month is not None:
        raise ValueError("Month cannot be specified when multiple years are provided.")
    
    if len(years) == 1 and month is not None:
        dataset_slug = f"realtimeqa-{years[0]}-{month:02d}"
    else:
        dataset_slug = "realtimeqa-" + "-".join(str(y) for y in years)
    
    return years, month, dataset_slug


def build_data_path(
    dataset_slug: str,
    model_slug: str,
    data_root: Path = Path("data"),
) -> Path:
    """
    Build path to model output directory.
    
    Parameters
    ----------
    dataset_slug : str
        Dataset identifier (e.g., "realtimeqa-2026-02")
    model_slug : str
        Model identifier (e.g., "allenai__OLMoE-1B-7B-0924-Instruct")
    data_root : Path, optional
        Root data directory (default: Path("data"))
    
    Returns
    -------
    Path
        Path to data/{dataset_slug}/{model_slug}/
    
    Examples
    --------
    >>> build_data_path("realtimeqa-2026-02", "allenai__OLMoE-1B-7B-0924-Instruct")
    PosixPath('data/realtimeqa-2026-02/allenai__OLMoE-1B-7B-0924-Instruct')
    """
    return data_root / dataset_slug / model_slug


def build_results_path(
    dataset_slug: str,
    model_slug: str,
    suffix: Optional[str] = None,
    data_root: Path = Path("data"),
) -> Path:
    """
    Build path to results parquet file.
    
    Parameters
    ----------
    dataset_slug : str
        Dataset identifier
    model_slug : str
        Model identifier
    suffix : str, optional
        Optional suffix for labeled results (e.g., "gemma3-1b" for
        results_labeled_gemma3-1b.parquet). If None, returns results.parquet
    data_root : Path, optional
        Root data directory (default: Path("data"))
    
    Returns
    -------
    Path
        Path to results file
    
    Examples
    --------
    >>> build_results_path("realtimeqa-2026-02", "allenai__OLMoE")
    PosixPath('data/realtimeqa-2026-02/allenai__OLMoE/results.parquet')
    
    >>> build_results_path("realtimeqa-2026-02", "allenai__OLMoE", "gemma3-1b")
    PosixPath('data/realtimeqa-2026-02/allenai__OLMoE/results_labeled_gemma3-1b.parquet')
    """
    data_path = build_data_path(dataset_slug, model_slug, data_root)
    
    if suffix:
        filename = f"results_labeled_{suffix}.parquet"
    else:
        filename = "results.parquet"
    
    return data_path / filename


def build_figures_path(
    script_name: str,
    dataset_slug: str,
    model_slug: str,
    figures_root: Path = Path("figures"),
) -> Path:
    """
    Build path to figures directory for a specific script.
    
    Parameters
    ----------
    script_name : str
        Name of the experiment script (e.g., "1.1-analyze-metrics")
    dataset_slug : str
        Dataset identifier
    model_slug : str
        Model identifier
    figures_root : Path, optional
        Root figures directory (default: Path("figures"))
    
    Returns
    -------
    Path
        Path to figures/{script_name}/{dataset_slug}/{model_slug}/
    
    Examples
    --------
    >>> build_figures_path("1.1-analyze-metrics", "realtimeqa-2026-02", "allenai__OLMoE")
    PosixPath('figures/1.1-analyze-metrics/realtimeqa-2026-02/allenai__OLMoE')
    """
    return figures_root / script_name / dataset_slug / model_slug
