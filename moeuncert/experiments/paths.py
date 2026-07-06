"""Path and slug resolution utilities for experiment scripts."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
#  Project root detection
# ---------------------------------------------------------------------------


def resolve_project_root() -> Path:
    """Return the project root directory (the parent of `moeuncert/`).

    This file lives inside `moeuncert/experiments/`, so the project root is
    three directories above it.
    """
    return Path(__file__).resolve().parent.parent.parent


def resolve_pretrained_models_dir(cache_root: Optional[Path] = None) -> Path:
    """Return the local model cache directory.

    Defaults to ``<project_root>/pretrained_models/``.  The pipeline also
    sets ``HF_HOME`` to this directory so that HuggingFace's auto-download
    and ``cache_dir`` both point to the same place.

    Parameters
    ----------
    cache_root : Path, optional
        Override directory.  If None, uses the default described above.

    Returns
    -------
    Path
        Absolute path to the local model cache directory.
    """
    if cache_root is None:
        cache_root = resolve_project_root() / "pretrained_models"
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def resolve_cache_dir(
    model_name: str,
    cache_root: Optional[Path] = None,
) -> Path:
    """Return the per-model cache directory inside the project.

    HuggingFace ``from_pretrained`` uses a nested ``models--<org>--<name>``
    layout.  We replicate that convention so the cache is a drop-in
    replacement for the global ``~/.cache/huggingface/hub/`` directory.

    Parameters
    ----------
    model_name : str
        Full HuggingFace model id, e.g. ``"google/gemma-4-26B-A4B-it"``.
    cache_root : Path, optional
        Override the root cache directory (default ``pretrained_models/``).

    Returns
    -------
    Path
        Absolute path that can be passed as ``cache_dir`` to
        ``AutoModel.from_pretrained(..., cache_dir=...)``.

    Examples
    --------
    >>> resolve_cache_dir("google/gemma-4-26B-A4B-it")
    PosixPath('/.../MoE-Uncertainty-Estimation/pretrained_models/models--google--gemma-4-26B-A4B-it')
    """
    root = resolve_pretrained_models_dir(cache_root)
    safe_name = model_name.replace("/", "--")
    return root / f"models--{safe_name}"


# ---------------------------------------------------------------------------
#  Model slug / dataset slug resolution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
#  Tokenizer loading (with 1.0's pad=eos reassignment)
# ---------------------------------------------------------------------------


def load_tokenizer_for_data(model_name: str, cache_dir: Optional[Path] = None):
    """Load a tokenizer and apply 1.0's ``pad_token = eos_token`` reassignment.

    ``1.0-generate-answers.py`` sets ``tokenizer.pad_token = tokenizer.eos_token``
    before generation so that batching works with models that lack a native
    pad token suitable for left-padding (e.g. OLMoE).  Consequently, the
    saved ``.pt`` batch files are padded with ``eos_token_id``, **not** the
    tokenizer's original ``pad_token_id``.

    Any script that loads those ``.pt`` files and passes ``pad_token_id`` to
    the collation functions (``read_and_collate_outputs``, ``_concat_parts``,
    ``load_model_outputs``) MUST use this helper — or manually apply the same
    reassignment — so that cross-batch padding matches the generation-time
    padding.  Otherwise ``find_generation_boundaries`` scans for the wrong pad
    token id and miscomputes the generation region, producing spurious
    empty-generation skips and "no pad token found" warnings.

    Parameters
    ----------
    model_name : str
        HuggingFace model id (e.g. ``"allenai/OLMoE-1B-7B-0924-Instruct"``).
    cache_dir : Path, optional
        Override the cache directory.  Defaults to ``resolve_cache_dir(model_name)``.

    Returns
    -------
    transformers.PreTrainedTokenizer
        Tokenizer with ``pad_token`` set to ``eos_token`` so that
        ``tokenizer.pad_token_id`` returns the eos id (matching the .pt files).
    """
    from transformers import AutoTokenizer

    if cache_dir is None:
        cache_dir = resolve_cache_dir(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
