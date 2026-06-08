"""
Data loading utilities for experiment scripts.

Provides functions for loading labeled datasets, model outputs, and multi-year
data across RealtimeQA experiments.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch

from .paths import resolve_model_slug, resolve_dataset_slug


def _parse_dataset_dir_name(name: str) -> Optional[Tuple[List[int], Optional[int]]]:
    """Parse a realtimeqa dataset directory name into years and month."""
    if not name.startswith("realtimeqa-"):
        return None

    parts = name.split("-")[1:]
    if not parts:
        return None

    month = None
    if len(parts) >= 2 and parts[-1].isdigit():
        candidate_month = int(parts[-1])
        if 1 <= candidate_month <= 12 and len(parts[-1]) == 2:
            month = candidate_month
            parts = parts[:-1]

    if not parts or any((not p.isdigit() or len(p) != 4) for p in parts):
        return None

    years = [int(p) for p in parts]
    if month is not None and len(years) != 1:
        return None

    return years, month


def _has_data_files(model_dir: Path) -> bool:
    """Check that a model data directory actually contains source data files."""
    return (
        any(model_dir.glob("results_labeled*.parquet"))
        or (model_dir / "results.parquet").exists()
        or any(model_dir.glob("base_generation/*.pt"))
        or any(model_dir.glob("evidence_generation/*.pt"))
    )


def _find_combined_dataset_dir(
    data_root: Path,
    model_slug: str,
    years: List[int],
    month: Optional[int],
) -> Optional[Tuple[Path, List[int]]]:
    """Find a dataset directory that covers the requested years."""
    candidates: List[Tuple[Path, List[int]]] = []
    for dataset_dir in data_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        parsed = _parse_dataset_dir_name(dataset_dir.name)
        if parsed is None:
            continue
        candidate_years, candidate_month = parsed
        if month is not None:
            exact_match = candidate_month == month and set(years).issubset(set(candidate_years))
            fallback_match = candidate_month is None and set(years).issubset(set(candidate_years))
            if not (exact_match or fallback_match):
                continue
        else:
            if candidate_month is not None:
                continue
            if not set(years).issubset(set(candidate_years)):
                continue
        model_dir = dataset_dir / model_slug
        if model_dir.exists() and _has_data_files(model_dir):
            candidates.append((model_dir, candidate_years))

    if not candidates:
        return None

    candidates.sort(key=lambda item: len(item[1]))
    return candidates[0]


def _filter_outputs_by_question_ids(
    outputs: Dict[str, torch.Tensor],
    question_ids: List[str],
) -> Dict[str, torch.Tensor]:
    """Filter model outputs to only the specified question IDs."""
    if "question_id" not in outputs:
        raise KeyError("outputs must contain 'question_id' for filtering.")

    allowed = {_normalize_question_id_value(qid) for qid in question_ids}
    output_qids = [
        _normalize_question_id_value(q)
        for q in _normalize_question_ids(outputs["question_id"])
    ]
    keep_indices = [idx for idx, qid in enumerate(output_qids) if qid in allowed]

    if len(keep_indices) == len(output_qids):
        return outputs

    idx_tensor = torch.tensor(keep_indices, dtype=torch.long)
    filtered: Dict[str, torch.Tensor] = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            filtered[key] = value.index_select(0, idx_tensor)
        elif isinstance(value, list):
            filtered[key] = [value[i] for i in keep_indices]
        else:
            filtered[key] = value
    return filtered


def _normalize_question_id_value(value) -> str:
    """Normalize a single question ID to a stable string form."""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith(".0"):
            candidate = stripped[:-2]
            if candidate.isdigit():
                return candidate
        # Handle tensor repr strings, e.g. "tensor(2025031414)"
        if stripped.startswith("tensor(") and stripped.endswith(")"):
            inner = stripped[len("tensor("):-1].split(",")[0].strip()
            if inner.isdigit():
                return inner
        return stripped
    return str(value)


def _normalize_question_ids(question_ids: List) -> List[str]:
    """Flatten and normalize question IDs to a list of strings."""
    flattened: List[str] = []
    for item in question_ids:
        if isinstance(item, list):
            flattened.extend(_normalize_question_id_value(qid) for qid in item)
        else:
            flattened.append(_normalize_question_id_value(item))
    return flattened


def _flatten_qid_tensors(question_ids: List) -> List:
    """Flatten any torch.Tensor values in a question_id list to plain Python ints.

    Production ``.pt`` files (written by ``experiments/1.0-generate-answers.py``
    with the HF dataset's ``set_format(type="torch")``) store
    ``question_id`` as a list of 0-d/1-d tensors (one tensor per batch
    file, each tensor holds the qids of that batch as int64s).  This
    helper turns that into a flat list of plain ints so the downstream
    normaliser sees ``[202401050, 202401051, ...]`` instead of
    ``[tensor([202401050, ...]), tensor([202405311, ...])]``.

    Handles:
      * 0-d tensor (``tensor(202401050)``) → ``[202401050]`` (single int)
      * 1-d tensor (``tensor([q0, q1, ...])``) → ``[q0, q1, ...]``
      * Nested lists of tensors / ints / strings → flattened
      * String elements, plain ints → passed through unchanged
    """
    flat: List = []
    for item in question_ids:
        if isinstance(item, torch.Tensor):
            # 0-d tensor: tolist() returns a single Python scalar (not
            # a list), so iterate it directly.  1-d tensor: tolist()
            # returns a list, iterate that.  n-d tensor: tolist() returns
            # nested lists, iterate the outer level.
            if item.dim() == 0:
                flat.append(int(item.item()))
            else:
                flat.extend(int(x) for x in item.tolist())
        elif isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _ensure_year_month_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure year/month columns exist using question_id as primary source.

    Falls back to question_date parsing only if question_id is unavailable.
    Using question_id avoids silent NaT failures from mixed date formats in
    Arrow-backed StringDtype columns (e.g. '2025/01/01' vs '2025-01-01').
    """
    if "year" in df.columns and "month" in df.columns:
        return df

    # Primary: derive from question_id (format is reliably YYYYMMDD)
    if "question_id" in df.columns:
        qid_str = df["question_id"].astype(str).str.replace(r"\D", "", regex=True)
        df = df.copy()
        if "year" not in df.columns:
            df["year"] = qid_str.str[:4].astype(int)
        if "month" not in df.columns:
            df["month"] = qid_str.str[4:6].astype(int)
        return df

    # Fallback: question_date parsing
    if "question_date" not in df.columns:
        return df

    dates = pd.to_datetime(df["question_date"], errors="coerce")
    if dates.isna().all():
        return df

    df = df.copy()
    if "year" not in df.columns:
        df["year"] = dates.dt.year
    if "month" not in df.columns:
        df["month"] = dates.dt.month
    return df


def _align_df_with_outputs(
    df: pd.DataFrame,
    outputs: Dict[str, torch.Tensor],
) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor]]:
    """Align labeled dataframe with available model outputs by question_id.

    Primary strategy: filter ``df`` to rows whose ``question_id`` appears
    in ``outputs["question_id"]`` (and filter ``outputs`` to keep only
    those qids, in the order they appear in ``df``).

    Fallback strategy: if qid sets do not overlap at all (e.g. the
    labeled parquet was regenerated with different data than the .pt
    files, or qid normalisation differs), fall back to *positional*
    alignment: assume ``outputs[i]`` corresponds to ``df.iloc[i]``.
    This is the safest assumption because both 1.0-generate-answers.py
    and 2.0-make-labels.py write rows in the same iteration order over
    the source HF dataset, so the i-th labeled row is the i-th model
    output *if and only if* both phases ran on the same data.  We log
    a clear WARNING showing the first few qids from each side so the
    user can spot inconsistencies at a glance.
    """
    if "question_id" not in outputs or "question_id" not in df.columns:
        return df, outputs

    if df.empty:
        raise ValueError("No labeled rows to align.")

    output_qids = {
        _normalize_question_id_value(q)
        for q in _normalize_question_ids(outputs["question_id"])
    }
    df = df.copy()
    df["question_id"] = df["question_id"].map(_normalize_question_id_value)
    df_qids = df["question_id"]
    missing_mask = ~df_qids.isin(output_qids)
    missing_count = int(missing_mask.sum())
    matched_count = len(df) - missing_count

    if matched_count == 0 and missing_count > 0:
        # Fall back to positional alignment.  This can happen when the
        # labeled parquet and the .pt files were generated from
        # different data (e.g. the user re-ran 2.0 with a newer
        # realtimeqa_original.parquet that has different questions than
        # what 1.0 used to generate the .pt files).  In that case, qid
        # intersection is empty but row ordering is still preserved.
        n_outputs = len(_normalize_question_ids(outputs["question_id"]))
        n_df = len(df)
        if n_outputs != n_df:
            raise ValueError(
                f"qid alignment failed: 0 qids match between labeled "
                f"dataframe ({n_df} rows) and model outputs ({n_outputs} "
                f"rows), and lengths differ so positional alignment is "
                f"unsafe.  Sample labeled qids: {df_qids.iloc[:3].tolist()}. "
                f"Sample output qids: "
                f"{_normalize_question_ids(outputs['question_id'])[:3]}. "
                f"Re-generate 1.0 / 2.0 with consistent data, or check "
                f"for stale .pt files in the data dir."
            )
        print(
            f"  WARNING: 0 qid matches between labeled dataframe and model "
            f"outputs ({n_df} rows each).  Falling back to POSITIONAL "
            f"alignment (outputs[i] ↔ df.iloc[i])."
        )
        print(
            f"    Sample labeled qids: {df_qids.iloc[:3].tolist()}"
        )
        print(
            f"    Sample output qids:  "
            f"{_normalize_question_ids(outputs['question_id'])[:3]}"
        )
        print(
            f"    If these are unrelated, regenerate 1.0 and 2.0 on "
            f"the same source data to fix alignment."
        )
        return df, outputs

    if missing_count:
        print(
            f"  WARNING: Dropping {missing_count} labeled rows without model outputs"
        )
        df = df.loc[~missing_mask].copy()

    outputs = _filter_outputs_by_question_ids(outputs, df["question_id"].tolist())
    return df, outputs


def load_labeled_dataset(
    data_dir: Path,
    label_model: Optional[str] = None,
) -> pd.DataFrame:
    """Load labeled parquet from 2.0-make-labels.py.

    Parameters
    ----------
    data_dir : Path
        Directory containing results (e.g., data/realtimeqa-2025-02/OLMoE-1B-7B-0924-Instruct/)
    label_model : str, optional
        If specified, loads results_labeled_{label_model}.parquet.
        Otherwise loads results_labeled.parquet.

    Returns
    -------
    pd.DataFrame
        Labeled dataset with hallucination labels.
    """
    if label_model:
        parquet_path = data_dir / f"results_labeled_{resolve_model_slug(label_model)}.parquet"
    else:
        parquet_path = data_dir / "results_labeled.parquet"

    if not parquet_path.exists():
        # Fallback to unlabeled results.parquet (for backward compatibility)
        parquet_path = data_dir / "results.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Labeled dataset not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"  Loaded: {parquet_path} ({len(df)} rows)")
    return df


def load_model_outputs(data_dir: Path) -> Dict[str, torch.Tensor]:
    """Load and collate model outputs from .pt batch files.

    Loads base_generation/ and evidence_generation/ directories separately so
    that an ``evidence_present`` boolean list can be added to the returned dict
    (False for base outputs, True for evidence/RAG outputs).  This allows
    downstream code to build composite ``(question_id, evidence_present)`` keys
    and avoid collisions when the same question has both modes.

    Parameters
    ----------
    data_dir : Path
        Directory containing base_generation/ and evidence_generation/ subdirs.

    Returns
    -------
    Dict[str, any]
        Collated outputs from all batches, including an ``evidence_present``
        list field.
    """
    from moeuncert.experiments.utils import read_and_collate_outputs

    base_dir = data_dir / "base_generation"
    evidence_dir = data_dir / "evidence_generation"

    parts: List[Dict] = []
    for d, is_evidence in [(base_dir, False), (evidence_dir, True)]:
        if not d.exists():
            continue
        files = sorted(d.glob("model_outputs__batch_*.pt"))
        if not files:
            continue
        part = read_and_collate_outputs(files, tokenizer=None, get_keys=None)
        if "question_id" in part:
            part["question_id"] = _normalize_question_ids(part["question_id"])
        # Determine entry count from question_id list or first tensor.
        n = len(part.get("question_id", []))
        if n == 0:
            for val in part.values():
                if isinstance(val, torch.Tensor):
                    n = val.shape[0]
                    break
        part["evidence_present"] = [is_evidence] * n
        parts.append(part)

    if not parts:
        raise FileNotFoundError(f"No batch output files found in {data_dir}")

    if len(parts) == 1:
        return parts[0]

    # Merge base and evidence parts.
    combined: Dict = {}
    for key in parts[0].keys():
        values = [p[key] for p in parts if key in p]
        if all(isinstance(v, torch.Tensor) for v in values):
            max_sizes = [max(v.shape[d] for v in values) for d in range(values[0].dim())]
            padded = []
            for t in values:
                pad_cfg = []
                for d in range(t.dim() - 1, 0, -1):
                    pad_cfg += [0, max_sizes[d] - t.shape[d]]
                padded.append(torch.nn.functional.pad(t, pad_cfg, value=0))
            combined[key] = torch.cat(padded, dim=0)
        else:
            merged: List = []
            for v in values:
                merged.extend(v)
            combined[key] = merged
        if key == "question_id":
            combined[key] = _normalize_question_ids(combined[key])
    return combined


def load_multi_year_data(
    data_root: Path,
    years: List[int],
    month: Optional[int],
    model: str,
    label_model: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor], Path]:
    """Load labeled data and model outputs across multiple years.

    Parameters
    ----------
    data_root : Path
        Root data directory.
    years : List[int]
        Years to load (e.g., [2022, 2023, 2024, 2025]).
    month : int, optional
        Month for single-year datasets (None = use all months per year).
    model : str
        Model name/slug.
    label_model : str, optional
        Label model name for labeled parquet filenames.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, torch.Tensor], Path]
        Concatenated labeled dataframe, collated model outputs, and the
        source directory where the data was actually loaded from.
    """
    model_slug = resolve_model_slug(model)
    all_dfs = []
    all_outputs_list = []

    print(f"\nLoading data for years: {years}")
    for year in years:
        _, _, dataset_slug = resolve_dataset_slug([year], month)
        data_dir = data_root / dataset_slug / model_slug

        if not data_dir.exists():
            print(f"  WARNING: {data_dir} not found, skipping year {year}")
            continue

        try:
            df = load_labeled_dataset(data_dir, label_model)
            df["year"] = year
            all_dfs.append(df)

            outputs = load_model_outputs(data_dir)
            all_outputs_list.append(outputs)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

    if not all_dfs:
        combined = _find_combined_dataset_dir(data_root, model_slug, years, month)
        if combined is not None:
            data_dir, dataset_years = combined
            print(
                f"  Using combined dataset: {data_dir.parent} "
                f"(covers years {dataset_years})"
            )
            df = load_labeled_dataset(data_dir, label_model)
            outputs = load_model_outputs(data_dir)

            if set(dataset_years) != set(years):
                df = _ensure_year_month_columns(df)
                if "year" not in df.columns:
                    raise ValueError(
                        f"Dataset {data_dir.parent} spans years {dataset_years}, but "
                        "the labeled parquet has no 'year' or 'question_date' column "
                        "to filter. Re-generate per-year data or provide a dataset "
                        "that matches the requested years."
                    )
                df = df[df["question_id"].apply(lambda x: str(x)[:4]).astype(int).isin(years)].copy()
                if month is not None:
                    if "month" not in df.columns:
                        df = _ensure_year_month_columns(df)
                    if "month" in df.columns:
                        df = df[df["month"].eq(month)].copy()
            elif "year" not in df.columns and len(years) == 1:
                df["year"] = years[0]

            if df.empty:
                raise ValueError(f"No data found for years {years}")

            df = df.reset_index(drop=True)
            df, outputs = _align_df_with_outputs(df, outputs)
            return df, outputs, data_dir

        raise ValueError(f"No data found for years {years}")

    # Concatenate dataframes
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined labeled data: {len(df_combined)} rows")

    # Concatenate outputs (question IDs are already unique across years)
    combined_outputs = {}
    for key in all_outputs_list[0].keys():
        if key == "question_id":
            # Flatten any nested lists: batch files may store question_id as
            # List[List[str]] so each extend() call could introduce a sub-list.
            all_qids: List[str] = []
            for o in all_outputs_list:
                for item in o[key]:
                    if isinstance(item, list):
                        all_qids.extend(item)
                    else:
                        all_qids.append(item)
            combined_outputs[key] = all_qids
        elif isinstance(all_outputs_list[0][key], torch.Tensor):
            combined_outputs[key] = torch.cat(
                [o[key] for o in all_outputs_list], dim=0
            )
        elif isinstance(all_outputs_list[0][key], list):
            # Extend all list-type fields (e.g., evidence_present) across years.
            merged_list: List = []
            for o in all_outputs_list:
                merged_list.extend(o[key])
            combined_outputs[key] = merged_list
        else:
            combined_outputs[key] = all_outputs_list[0][key]

    # Sanity-check: question_id length must match the first tensor batch dimension.
    if "question_id" in combined_outputs:
        first_tensor_key = next(
            (k for k in combined_outputs if isinstance(combined_outputs[k], torch.Tensor)),
            None,
        )
        if first_tensor_key is not None:
            n_tensor = combined_outputs[first_tensor_key].shape[0]
            n_qids = len(combined_outputs["question_id"])
            assert n_qids == n_tensor, (
                f"question_id length ({n_qids}) does not match "
                f"tensor batch size ({n_tensor})"
            )

    df_combined, combined_outputs = _align_df_with_outputs(df_combined, combined_outputs)
    # Use the first year's directory as the canonical source dir
    first_year_data_dir = data_root / resolve_dataset_slug([years[0]], month)[2] / model_slug
    return df_combined, combined_outputs, first_year_data_dir


def load_sampled_outputs(
    data_root: Path,
    years: List[int],
    month: Optional[int],
    model: str,
    num_samples: int = 5,
) -> Dict[str, List]:
    """Load sampled generation outputs across multiple years.

    Reads .pt batch files from sampled_generation/ directories and collates
    them into per-question data structures needed by SemanticUncertainty and
    SemanticEnergy baselines.

    Parameters
    ----------
    data_root : Path
        Root data directory.
    years : List[int]
        Years to load.
    month : int, optional
        Month for single-year datasets.
    model : str
        Model name/slug.
    num_samples : int
        Number of sampled responses per question (default: 5).

    Returns
    -------
    Dict with keys:
        responses_by_qid : Dict[str, List[str]]
            Per-question list of sampled response strings.
        log_probs_by_qid : Dict[str, List[List[float]]]
            Per-question per-response per-token log probabilities.
        logits_by_qid : Dict[str, List[List[float]]]
            Per-question per-response per-token logits (for SemanticEnergy).
        labels_by_qid : Dict[str, int]
            Per-question hallucination label (from the base labeled dataset).
    """
    from .utils import read_and_collate_outputs

    model_slug = resolve_model_slug(model)
    all_responses_by_qid: Dict[str, List[str]] = {}
    all_logprobs_by_qid: Dict[str, List[List[float]]] = {}
    all_logits_by_qid: Dict[str, List[List[float]]] = {}
    found_any = False

    print(f"\nLoading sampled outputs for years: {years}")
    combined = None
    for year in years:
        _, _, dataset_slug = resolve_dataset_slug([year], month)
        sampled_dir = data_root / dataset_slug / model_slug / "sampled_generation"

        if not sampled_dir.exists():
            print(f"  WARNING: {sampled_dir} not found, skipping year {year}")
            continue

        batch_files = sorted(sampled_dir.glob("sampled_outputs__batch_*.pt"))
        if not batch_files:
            print(f"  WARNING: No batch files in {sampled_dir}")
            continue
        found_any = True

        # Read and collate all batches
        outputs = read_and_collate_outputs(
            batch_files, tokenizer=None, get_keys=None
        )
        print(f"  Loaded {len(batch_files)} batch files from year {year}")

        # Extract question_ids (list of strings)
        qids = outputs["question_id"]

        # For each question, collect sampled responses and log probs
        for idx, qid in enumerate(qids):
            qid = _normalize_question_id_value(qid)

            # Filter by requested years when loading from combined dataset
            if combined is not None:
                qid_year = int(qid[:4]) if len(qid) >= 4 and qid[:4].isdigit() else None
                if qid_year is not None and qid_year not in years:
                    continue

            # Collect responses from each sample
            responses = []
            log_probs = []
            logits = []

            for s in range(num_samples):
                seq_key = f"sequences_sample{s}"
                ll_key = f"log_likelihoods_sample{s}"
                scores_key = f"scores_sample{s}"

                if seq_key not in outputs or ll_key not in outputs:
                    continue

                # Decode the generated sequence (outputs include full seq,
                # we need only the generated part)
                gen_tokens = outputs[seq_key][idx]
                # Filter out padding (token_id == 0 or pad_token)
                gen_tokens = gen_tokens[gen_tokens != 0]

                # Get the response text — we need tokenizer for this,
                # so store token IDs and convert later, or skip text for now.
                # Store token IDs temporarily; calling code will decode if needed.
                response_tokens = gen_tokens.tolist()
                responses.append(response_tokens)

                # Log-likelihoods: (gen_seq_len,) per response
                ll = outputs[ll_key][idx].tolist()
                log_probs.append(ll)

                # Raw scores/logits: (gen_seq_len, vocab_size) per response
                if scores_key in outputs:
                    scores = outputs[scores_key][idx]  # (gen_seq_len, vocab_size)
                    # Extract logits for the generated tokens
                    gen_len = min(len(ll), scores.shape[0])
                    if gen_len > 0 and len(gen_tokens) > 0:
                        # Get logits at each position for the actual generated token
                        token_ids = gen_tokens[:gen_len]
                        response_logits = scores[:gen_len].gather(
                            -1, token_ids.unsqueeze(-1).to(scores.device)
                        ).squeeze(-1).tolist()
                        logits.append(response_logits)

            if responses:
                all_responses_by_qid[qid] = responses
                all_logprobs_by_qid[qid] = log_probs
                if logits:
                    all_logits_by_qid[qid] = logits

    if not found_any:
        combined = _find_combined_dataset_dir(data_root, model_slug, years, month)
        if combined is not None:
            data_dir, dataset_years = combined
            sampled_dir = data_dir / "sampled_generation"
            if sampled_dir.exists():
                print(
                    f"  Using combined dataset: {data_dir.parent} "
                    f"(covers years {dataset_years})"
                )
                batch_files = sorted(sampled_dir.glob("sampled_outputs__batch_*.pt"))
                if batch_files:
                    outputs = read_and_collate_outputs(
                        batch_files, tokenizer=None, get_keys=None
                    )
                    print(
                        f"  Loaded {len(batch_files)} batch files from combined dataset"
                    )
                    qids = outputs["question_id"]
                    for idx, qid in enumerate(qids):
                        qid = _normalize_question_id_value(qid)

                        # Filter by requested years when loading from combined dataset
                        qid_year = int(qid[:4]) if len(qid) >= 4 and qid[:4].isdigit() else None
                        if qid_year is not None and qid_year not in years:
                            continue

                        responses = []
                        log_probs = []
                        logits = []

                        for s in range(num_samples):
                            seq_key = f"sequences_sample{s}"
                            ll_key = f"log_likelihoods_sample{s}"
                            scores_key = f"scores_sample{s}"

                            if seq_key not in outputs or ll_key not in outputs:
                                continue

                            gen_tokens = outputs[seq_key][idx]
                            gen_tokens = gen_tokens[gen_tokens != 0]

                            response_tokens = gen_tokens.tolist()
                            responses.append(response_tokens)

                            ll = outputs[ll_key][idx].tolist()
                            log_probs.append(ll)

                            if scores_key in outputs:
                                scores = outputs[scores_key][idx]
                                gen_len = min(len(ll), scores.shape[0])
                                if gen_len > 0 and len(gen_tokens) > 0:
                                    token_ids = gen_tokens[:gen_len]
                                    response_logits = scores[:gen_len].gather(
                                        -1, token_ids.unsqueeze(-1).to(scores.device)
                                    ).squeeze(-1).tolist()
                                    logits.append(response_logits)

                        if responses:
                            all_responses_by_qid[qid] = responses
                            all_logprobs_by_qid[qid] = log_probs
                            if logits:
                                all_logits_by_qid[qid] = logits

    print(f"  Total questions with sampled data: {len(all_responses_by_qid)}")
    return {
        "responses_by_qid": all_responses_by_qid,
        "log_probs_by_qid": all_logprobs_by_qid,
        "logits_by_qid": all_logits_by_qid,
    }


# ---------------------------------------------------------------------------
# Streaming variants (memory-bounded alternatives to the loaders above)
# ---------------------------------------------------------------------------
# These functions exist to support scripts (3.1, 3.2) that need to iterate
# over batch .pt files without materialising the full concatenated dict in
# host RAM.  They yield per-batch (or per-year) data; the caller accumulates
# into running aggregates.
#
# The legacy ``load_multi_year_data`` and ``load_sampled_outputs`` are NOT
# changed; scripts that already work (3.0, 4.0, 5.0, 0.1, 0.2) keep using
# them.  Only 3.1 and 3.2 migrate to the streaming path.


def _stream_year_outputs(
    data_dir: Path,
    filter_keys: Optional[Set[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield (base_outputs, evidence_outputs) dicts in turn, then any extras.

    Mirrors ``load_model_outputs``'s base/evidence split, but yields one
    (base, evidence) pair per *batch* rather than concatenating across
    batches.  ``filter_keys`` is forwarded to ``iter_batch_outputs`` so
    unwanted keys (e.g. expert_hidden_scores when LLM-Check is not
    being computed) are dropped before the next batch is read.
    """
    from .utils import iter_batch_outputs

    base_dir = data_dir / "base_generation"
    evidence_dir = data_dir / "evidence_generation"

    base_files = sorted(base_dir.glob("model_outputs__batch_*.pt")) if base_dir.exists() else []
    evidence_files = sorted(evidence_dir.glob("model_outputs__batch_*.pt")) if evidence_dir.exists() else []

    if not base_files and not evidence_files:
        return

    max_len = max(len(base_files), len(evidence_files))
    for i in range(max_len):
        if i < len(base_files):
            base_batch = next(iter_batch_outputs([base_files[i]], filter_keys))
            base_batch["evidence_present"] = [False] * _batch_len(base_batch)
            yield base_batch
        if i < len(evidence_files):
            evidence_batch = next(iter_batch_outputs([evidence_files[i]], filter_keys))
            evidence_batch["evidence_present"] = [True] * _batch_len(evidence_batch)
            yield evidence_batch


def _batch_len(batch: Dict[str, Any]) -> int:
    """Return the per-row count of a batch dict."""
    if "question_id" in batch:
        return len(batch["question_id"])
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            return v.shape[0]
    return 0


def stream_multi_year_data(
    data_root: Path,
    years: List[int],
    month: Optional[int],
    model: str,
    label_model: Optional[str] = None,
    filter_keys: Optional[Set[str]] = None,
) -> Iterator[Tuple[pd.DataFrame, Dict[str, torch.Tensor], Path]]:
    """Yield (per_year_df, per_year_outputs, data_dir) for each year with data.

    Streaming counterpart to ``load_multi_year_data``.  Per-year model
    outputs are filtered to ``filter_keys`` (if provided) and *not*
    concatenated across years.  The caller is responsible for
    accumulating into running aggregates.

    Mirrors the legacy loader's year-discovery logic exactly:
      * Per-year dir (``data/realtimeqa-YYYY/.../``) if it exists.
      * Otherwise fall back to a combined dir that covers the year
        (``data/realtimeqa-Y1-Y2-.../.../``).
      * Within a year, base and evidence batch files are read
        independently and yielded in order.

    The ``pd.DataFrame`` yielded alongside each year's outputs is the
    per-year labeled subset (with a ``year`` column set).  Callers
    typically ``pd.concat`` the per-year dataframes at the end.
    """
    model_slug = resolve_model_slug(model)

    print(f"\nLoading data for years (streaming): {years}")
    yielded = 0
    for year in years:
        _, _, dataset_slug = resolve_dataset_slug([year], month)
        data_dir = data_root / dataset_slug / model_slug

        if not data_dir.exists():
            print(f"  WARNING: {data_dir} not found, skipping year {year}")
            continue

        try:
            df = load_labeled_dataset(data_dir, label_model)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        # Per-year outputs: collate per-batch via streaming helper, then
        # concatenate within the year (one year at a time, so peak RAM
        # is bounded by the largest single year rather than the whole
        # corpus).  This matches the legacy load_model_outputs' behaviour
        # of concatenating base+evidence within a data_dir, while
        # *not* concatenating across years.
        year_outputs = _load_year_outputs_streaming(data_dir, filter_keys)
        if not year_outputs:
            continue

        df["year"] = year
        df, year_outputs = _align_df_with_outputs(df, year_outputs)
        yielded += 1
        yield df, year_outputs, data_dir

    if yielded == 0:
        # Combined-dir fallback (mirrors legacy line 384-417)
        combined = _find_combined_dataset_dir(data_root, model_slug, years, month)
        if combined is not None:
            data_dir, dataset_years = combined
            print(
                f"  Using combined dataset: {data_dir.parent} "
                f"(covers years {dataset_years})"
            )
            df = load_labeled_dataset(data_dir, label_model)
            year_outputs = _load_year_outputs_streaming(data_dir, filter_keys)
            if not year_outputs:
                raise ValueError(f"No data found for years {years}")

            if set(dataset_years) != set(years):
                df = _ensure_year_month_columns(df)
                if "year" not in df.columns:
                    raise ValueError(
                        f"Dataset {data_dir.parent} spans years {dataset_years}, but "
                        "the labeled parquet has no 'year' or 'question_date' column "
                        "to filter. Re-generate per-year data or provide a dataset "
                        "that matches the requested years."
                    )
                df = df[df["question_id"].apply(lambda x: str(x)[:4]).astype(int).isin(years)].copy()
                if month is not None:
                    if "month" not in df.columns:
                        df = _ensure_year_month_columns(df)
                    if "month" in df.columns:
                        df = df[df["month"].eq(month)].copy()
            elif "year" not in df.columns and len(years) == 1:
                df["year"] = years[0]

            if df.empty:
                raise ValueError(f"No data found for years {years}")

            df = df.reset_index(drop=True)
            df, year_outputs = _align_df_with_outputs(df, year_outputs)
            yield df, year_outputs, data_dir
            return

        raise ValueError(f"No data found for years {years}")


def _load_year_outputs_streaming(
    data_dir: Path,
    filter_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Concatenate base+evidence batch files for a single year, but stream
    each batch through ``iter_batch_outputs`` to drop unwanted keys early.

    Equivalent to ``load_model_outputs(data_dir)`` but with optional
    ``filter_keys`` for memory savings between batches.

    Each .pt file is saved by ``experiments/1.0-generate-answers.py`` in
    the "dict of lists of tensors" format: ``{key: [tensor_batch_0,
    tensor_batch_1, ...]}``.  ``iter_batch_outputs`` yields the loaded
    dict for each file; we then ``extend`` (not ``append``) so the
    per-key list flattens correctly.
    """
    from .utils import iter_batch_outputs

    base_dir = data_dir / "base_generation"
    evidence_dir = data_dir / "evidence_generation"

    base_files = sorted(base_dir.glob("model_outputs__batch_*.pt")) if base_dir.exists() else []
    evidence_files = sorted(evidence_dir.glob("model_outputs__batch_*.pt")) if evidence_dir.exists() else []

    if not base_files and not evidence_files:
        return {}

    parts: List[Dict[str, Any]] = []
    for files, is_evidence in [(base_files, False), (evidence_files, True)]:
        if not files:
            continue
        part: Dict[str, Any] = {}
        for loaded in iter_batch_outputs(files, filter_keys):
            # Each loaded dict is in production format: {key: [tensor, tensor, ...]}.
            # Extend (not append) so part[key] becomes a flat list of tensors.
            for k, v in loaded.items():
                if isinstance(v, list):
                    if k not in part:
                        part[k] = []
                    part[k].extend(v)
                else:
                    if k not in part:
                        part[k] = []
                    part[k].append(v)
        if "question_id" in part:
            # Production .pt files store question_id as a list of
            # 0-d/1-d tensors (one per batch, each tensor = qids of that
            # batch).  ``_normalize_question_ids`` would otherwise
            # stringify the whole tensor (e.g. ``"tensor([N, N, ...])"``)
            # which never matches the labeled parquet.  Flatten any
            # tensor values to plain ints first.
            part["question_id"] = _flatten_qid_tensors(part["question_id"])
            part["question_id"] = _normalize_question_ids(part["question_id"])
        n = len(part.get("question_id", []))
        if n == 0:
            for val in part.values():
                if isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
                    n = val[0].shape[0]
                    break
        part["evidence_present"] = [is_evidence] * n
        parts.append(part)

    if not parts:
        return {}

    if len(parts) == 1:
        return _concat_parts(parts)

    return _concat_parts(parts)


def _concat_parts(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate a list of per-year/per-mode dicts into one.

    Each input ``part`` is in the production format
    ``{key: [tensor_batch_0, tensor_batch_1, ...]}``.  We flatten
    across parts (so part A's batches come before part B's batches) and
    then ``torch.cat`` the tensor values, padding to max size in each
    dim (mirrors ``read_and_collate_outputs``).

    Iterates the *union* of all keys across parts (not just
    ``parts[0].keys()``) so that evidence-only or base-only keys
    (e.g. ``perplexity`` if ``return_baseline_features`` was set for
    one phase but not the other) are preserved.
    """
    combined: Dict[str, Any] = {}
    all_keys: set = set()
    for p in parts:
        all_keys.update(p.keys())
    for key in sorted(all_keys):
        # Flatten the per-part lists into a single list of tensors.
        flat: List = []
        for p in parts:
            if key in p:
                v = p[key]
                if isinstance(v, list):
                    flat.extend(v)
                else:
                    flat.append(v)
        if not flat:
            continue
        if all(isinstance(v, torch.Tensor) for v in flat):
            if len(flat) == 1 and flat[0].dim() == 0:
                combined[key] = flat[0]
                continue
            if all(v.shape == flat[0].shape for v in flat):
                combined[key] = torch.concat(flat, dim=0)
            else:
                # Pad to max size in each dim before concat.
                max_sizes = [max(v.shape[d] for v in flat) for d in range(flat[0].dim())]
                padded = []
                for t in flat:
                    pad_cfg = []
                    for d in range(t.dim() - 1, 0, -1):
                        pad_cfg += [0, max_sizes[d] - t.shape[d]]
                    padded.append(torch.nn.functional.pad(t, pad_cfg, value=0))
                combined[key] = torch.cat(padded, dim=0)
        else:
            combined[key] = flat
    if "question_id" in combined:
        combined["question_id"] = _normalize_question_ids(combined["question_id"])
    return combined


def _extract_qid_samples(
    batch: Dict[str, Any],
    num_samples: int,
    requested_years: Optional[List[int]] = None,
) -> Tuple[Dict[str, List], List[str]]:
    """Extract per-question sampled outputs from a single batch dict.

    Returns (accumulator_updates, qids_seen) where accumulator_updates
    is a dict that can be merged into a global {qid: [samples]} dict
    via ``.extend()`` on each per-qid list.

    The streaming version of ``load_sampled_outputs`` calls this once
    per yielded batch; the caller accumulates across batches.

    Contract for ``batch`` (normalised by ``_normalize_sampled_batch``):
      * ``question_id``: flat list of N strings
      * ``sequences_sampleS``: 2-d long tensor of shape (N, seq_len)
      * ``log_likelihoods_sampleS``: 2-d tensor of shape (N, gen_len)
      * ``scores_sampleS``: 3-d tensor of shape (N, gen_len, vocab)

    For defensive safety, ``question_id`` is normalised internally
    to a flat list of strings in case the caller skipped
    ``_normalize_sampled_batch`` (e.g. legacy callers).
    """
    qids_raw = batch.get("question_id", [])
    if not isinstance(qids_raw, list):
        qids_raw = [qids_raw]
    qids = _normalize_question_ids(_flatten_qid_tensors(qids_raw))
    updates: Dict[str, List] = {}

    for idx, qid in enumerate(qids):
        if requested_years is not None:
            qid_year = int(qid[:4]) if len(qid) >= 4 and qid[:4].isdigit() else None
            if qid_year is not None and qid_year not in requested_years:
                continue

        responses: List[List[int]] = []
        log_probs: List[List[float]] = []
        logits: List[List[float]] = []

        for s in range(num_samples):
            seq_key = f"sequences_sample{s}"
            ll_key = f"log_likelihoods_sample{s}"
            scores_key = f"scores_sample{s}"

            if seq_key not in batch or ll_key not in batch:
                continue

            gen_tokens = batch[seq_key][idx]
            gen_tokens = gen_tokens[gen_tokens != 0]
            response_tokens = gen_tokens.tolist()
            responses.append(response_tokens)

            ll = batch[ll_key][idx].tolist()
            log_probs.append(ll)

            if scores_key in batch:
                scores = batch[scores_key][idx]
                gen_len = min(len(ll), scores.shape[0])
                if gen_len > 0 and len(gen_tokens) > 0:
                    token_ids = gen_tokens[:gen_len]
                    response_logits = scores[:gen_len].gather(
                        -1, token_ids.unsqueeze(-1).to(scores.device)
                    ).squeeze(-1).tolist()
                    logits.append(response_logits)

        if responses:
            # STREAMING-FIX: use .extend() (not =) so that a question_id
            # appearing in multiple batches accumulates all its samples
            # rather than being clobbered by the last batch.  The legacy
            # load_sampled_outputs retains its original behaviour
            # (clobber) for backwards compatibility; only the streaming
            # path fixes this.  Documented in the PR description.
            if qid not in updates:
                updates[qid] = {"responses": [], "log_probs": [], "logits": []}
            updates[qid]["responses"].extend(responses)
            updates[qid]["log_probs"].extend(log_probs)
            updates[qid]["logits"].extend(logits)

    return updates, qids


def _normalize_sampled_batch(loaded: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a single .pt file's contents to the shape
    ``_extract_qid_samples`` expects.

    Production sampled_generation .pt files come in two shapes:

    Format A (1.0-style): ``{question_id: [t_b0], sequences_sample0:
      [tensor_b0], ...}`` where ``t_b0`` is a 1-d int64 tensor of N
      qids, and ``tensor_b0`` is a 2-d tensor of shape ``(N, ...)``.

    Format B (1.1-style, the user's data): ``{question_id: [q0, q1,
      ...], sequences_sample0: [tensor_b0], ...}`` where
      ``question_id`` is a flat list of N items (each a 0-d int64
      tensor or plain int) and ``tensor_b0`` is a 2-d tensor of
      shape ``(N, ...)``.

    We normalise to:
      * ``question_id``: flat list of N plain ints/strings.
      * ``<tensor_key>``: 2-d (or higher-d) tensor of shape ``(N, ...)``.

    The outer list wrapper is unwrapped (one .pt file = one saved
    batch, so the list always has length 1 in production).
    """
    normalised: Dict[str, Any] = {}
    for k, v in loaded.items():
        if k == "question_id":
            # Could be Format A ``[1-d tensor]`` or Format B
            # ``[q0, q1, ...]``.  Unwrap the list, then flatten tensors.
            if isinstance(v, list):
                inner = _flatten_qid_tensors(v)
            else:
                inner = _flatten_qid_tensors([v])
            normalised[k] = _normalize_question_ids(inner)
        elif isinstance(v, list):
            # Tensor keys: list of 1 tensor (one saved batch per file).
            # Just take the first tensor; if the file ever has multiple
            # saved batches, we'd need to iterate, but production doesn't.
            if len(v) == 1:
                normalised[k] = v[0]
            elif len(v) > 1:
                # Defensive: concat along dim 0 to handle multi-batch files.
                normalised[k] = torch.cat(v, dim=0)
            else:
                normalised[k] = v
        else:
            normalised[k] = v
    return normalised


def stream_sampled_outputs(
    data_root: Path,
    years: List[int],
    month: Optional[int],
    model: str,
    num_samples: int = 5,
    filter_keys: Optional[Set[str]] = None,
) -> Dict[str, List]:
    """Streaming counterpart to ``load_sampled_outputs``.

    Yields zero or more dicts of the same shape as the legacy return
    value, but only the final yielded dict contains the full
    accumulated state.  Use ``next(iter(stream_sampled_outputs(...)))``
    to get the final result in a single call, matching the legacy
    call pattern.

    The streaming aspect is on the disk-read side (one batch at a
    time) and on the per-batch filter side (drop unused sample keys
    early).  Peak RAM is bounded by one batch at a time plus the
    final accumulated dict (which is the same as the legacy
    function's peak).

    The streaming path also fixes a latent qid-clobbering bug in
    the legacy function (where a question_id appearing in multiple
    batches would be overwritten by the last batch's data instead
    of having its samples accumulated).  See
    ``_extract_qid_samples`` for the fix.

    Per-file format note: sampled_generation .pt files (written by
    ``1.1-generate-baseline-samples.py``) have
    ``{question_id: [q0, q1, ...], sequences_sample0: [tensor_b0],
    log_likelihoods_sample0: [tensor_b0], ...}`` — i.e. ``question_id``
    is a list of N items (one per question in the batch) but tensor
    keys have a list of M=1 tensors (one tensor per saved batch, since
    one .pt file = one saved batch).  In contrast, base+evidence .pt
    files (1.0) have ``{question_id: [tensor_b0], ...}`` where the
    tensor itself has shape ``(N,)`` (N = questions in the batch).  We
    handle both formats in ``_normalize_sampled_batch``.
    """
    from .utils import iter_batch_outputs

    model_slug = resolve_model_slug(model)
    all_responses_by_qid: Dict[str, List[List[int]]] = {}
    all_logprobs_by_qid: Dict[str, List[List[float]]] = {}
    all_logits_by_qid: Dict[str, List[List[float]]] = {}
    found_any = False

    print(f"\nLoading sampled outputs for years (streaming): {years}")
    for year in years:
        _, _, dataset_slug = resolve_dataset_slug([year], month)
        sampled_dir = data_root / dataset_slug / model_slug / "sampled_generation"

        if not sampled_dir.exists():
            print(f"  WARNING: {sampled_dir} not found, skipping year {year}")
            continue

        batch_files = sorted(sampled_dir.glob("sampled_outputs__batch_*.pt"))
        if not batch_files:
            print(f"  WARNING: No batch files in {sampled_dir}")
            continue
        found_any = True

        print(f"  Streaming {len(batch_files)} batch files from year {year}")
        for loaded in iter_batch_outputs(batch_files, filter_keys):
            # Normalise to the shape ``_extract_qid_samples`` expects.
            # See ``_normalize_sampled_batch`` for the per-key rules.
            single_batch = _normalize_sampled_batch(loaded)
            updates, _ = _extract_qid_samples(
                single_batch, num_samples, requested_years=years
            )
            for qid, parts in updates.items():
                if qid not in all_responses_by_qid:
                    all_responses_by_qid[qid] = []
                    all_logprobs_by_qid[qid] = []
                    all_logits_by_qid[qid] = []
                all_responses_by_qid[qid].extend(parts["responses"])
                all_logprobs_by_qid[qid].extend(parts["log_probs"])
                all_logits_by_qid[qid].extend(parts["logits"])

    if not found_any:
        combined = _find_combined_dataset_dir(data_root, model_slug, years, month)
        if combined is not None:
            data_dir, dataset_years = combined
            sampled_dir = data_dir / "sampled_generation"
            if sampled_dir.exists():
                print(
                    f"  Using combined dataset: {data_dir.parent} "
                    f"(covers years {dataset_years})"
                )
                batch_files = sorted(sampled_dir.glob("sampled_outputs__batch_*.pt"))
                if batch_files:
                    for loaded in iter_batch_outputs(batch_files, filter_keys):
                        single_batch = _normalize_sampled_batch(loaded)
                        updates, _ = _extract_qid_samples(
                            single_batch, num_samples, requested_years=years
                        )
                        for qid, parts in updates.items():
                            if qid not in all_responses_by_qid:
                                all_responses_by_qid[qid] = []
                                all_logprobs_by_qid[qid] = []
                                all_logits_by_qid[qid] = []
                            all_responses_by_qid[qid].extend(parts["responses"])
                            all_logprobs_by_qid[qid].extend(parts["log_probs"])
                            all_logits_by_qid[qid].extend(parts["logits"])

    print(f"  Total questions with sampled data: {len(all_responses_by_qid)}")
    return {
        "responses_by_qid": all_responses_by_qid,
        "log_probs_by_qid": all_logprobs_by_qid,
        "logits_by_qid": all_logits_by_qid,
    }
