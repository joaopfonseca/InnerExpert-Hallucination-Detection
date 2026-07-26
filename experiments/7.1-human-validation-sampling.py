"""
7.1 Human-validation sampling: export 200 random rows (25 per model per 2x2
contingency cell) from the labelled RealtimeQA training data for manual
annotation.

The spreadsheet contains one row per sample with question, evidence,
generated answer (with inline ``[SPAN]...[/SPAN]`` markers around
LLM-flagged hallucinated spans), both weak and LLM labels, and empty
columns for the human annotator to fill.

Sampling design (full 2x2 contingency)::

    Cell                         weak   llm
    --------------------------- ----- -----
    agree_grounded                 0     0
    agree_hallucinated             1     1
    disagree_weak_hall_llm_grounded 1     0
    disagree_weak_grounded_llm_hall 0     1

25 rows per cell x 2 models x 4 cells = 200 rows.

Usage::

    python experiments/7.1-human-validation-sampling.py

Paths to the labelled parquet files are hardcoded relative to the project
root (same convention as 7.0).  Output is written to
``analysis/human_validation_sample.xlsx``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

# ---------------------------------------------------------------------------
# Paths (hardcoded, relative to project root — same as 7.0)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "realtimeqa-2024-2025"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

MODEL_PARQUETS = {
    "OLMoE-1B-7B": (
        DATA_DIR
        / "allenai__OLMoE-1B-7B-0924-Instruct"
        / "results_labeled_zai-org__GLM-5.1.parquet"
    ),
    "Gemma-4-26B": (
        DATA_DIR
        / "google__gemma-4-26B-A4B-it"
        / "results_labeled_zai-org__GLM-5.1.parquet"
    ),
}

N_PER_CELL = 25
SEED = 42

# (cell_name, weak_label, llm_label)
CELLS: List[Tuple[str, int, int]] = [
    ("agree_grounded", 0, 0),
    ("agree_hallucinated", 1, 1),
    ("disagree_weak_hall_llm_grounded", 1, 0),
    ("disagree_weak_grounded_llm_hall", 0, 1),
]


# ---------------------------------------------------------------------------
# Span marking (inline [SPAN]...[/SPAN])
# ---------------------------------------------------------------------------
def mark_spans_inline(answer: str, spans: Optional[object]) -> str:
    """Return *answer* with each span wrapped in ``[SPAN]...[/SPAN]``.

    Spans are treated as exact substrings.  Overlapping spans are resolved
    by preferring the longer match.  Spans not found in *answer* are
    silently skipped.
    """
    if spans is None:
        return answer

    # Normalise spans into a list of non-empty strings
    if isinstance(spans, float) and np.isnan(spans):
        return answer
    if isinstance(spans, (list, tuple, np.ndarray)):
        span_list = [str(s) for s in spans if str(s).strip()]
    else:
        span_list = [str(spans)] if str(spans).strip() else []

    if not span_list:
        return answer

    # Find all match intervals
    intervals: List[Tuple[int, int]] = []
    for span in span_list:
        start = 0
        while True:
            idx = answer.find(span, start)
            if idx == -1:
                break
            intervals.append((idx, idx + len(span)))
            start = idx + 1

    if not intervals:
        return answer

    # Sort by start, then by -length (longer first)
    intervals.sort(key=lambda iv: (iv[0], -(iv[1] - iv[0])))

    # Resolve overlaps: greedily pick non-overlapping intervals
    merged: List[Tuple[int, int]] = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # Build output
    parts: List[str] = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            parts.append(answer[cursor:start])
        parts.append("[SPAN]" + answer[start:end] + "[/SPAN]")
        cursor = end
    if cursor < len(answer):
        parts.append(answer[cursor:])

    return "".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_labeled_data() -> Dict[str, pd.DataFrame]:
    """Load both models' labelled parquet files."""
    dfs = {}
    for name, path in MODEL_PARQUETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Labelled parquet not found: {path}")
        df = pd.read_parquet(path)
        df["evidence_present"] = df["evidence_present"].astype(int)
        df["label_llm_answer"] = df["label_llm_answer"].astype(float)
        dfs[name] = df
        print(f"  {name}: {len(df)} rows from {path.name}")
    return dfs


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_2x2_cells(
    df: pd.DataFrame,
    model_name: str,
    n_per_cell: int,
    seed: int,
) -> pd.DataFrame:
    """Sample *n_per_cell* rows from each 2x2 contingency cell.

    Rows with NaN ``label_llm_answer`` are dropped before sampling.
    If a cell has fewer than *n_per_cell* rows, all available are taken
    and a warning is printed.
    """
    rng = np.random.default_rng(seed)

    # Drop rows where LLM label is NaN (failed API calls)
    df_clean = df.dropna(subset=["label_llm_answer"]).copy()
    df_clean["label_llm_answer"] = df_clean["label_llm_answer"].astype(int)
    df_clean["label_weak_hallucination"] = df_clean[
        "label_weak_hallucination"
    ].astype(int)
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        print(f"  {model_name}: dropped {n_dropped} rows with NaN LLM label")

    sampled_parts: List[pd.DataFrame] = []
    for cell_name, weak_label, llm_label in CELLS:
        mask = (
            (df_clean["label_weak_hallucination"] == weak_label)
            & (df_clean["label_llm_answer"] == llm_label)
        )
        cell_df = df_clean[mask]
        n_available = len(cell_df)

        if n_available == 0:
            print(
                f"  WARNING: {model_name} / {cell_name}: 0 rows available, skipping"
            )
            continue

        n_take = min(n_per_cell, n_available)
        if n_take < n_per_cell:
            print(
                f"  WARNING: {model_name} / {cell_name}: only {n_available} rows "
                f"available (requested {n_per_cell})"
            )

        indices = rng.choice(cell_df.index.to_numpy(), size=n_take, replace=False)
        cell_sample = cell_df.loc[indices].copy()
        cell_sample["cell"] = cell_name
        sampled_parts.append(cell_sample)
        print(f"  {model_name} / {cell_name}: sampled {n_take}/{n_available}")

    if not sampled_parts:
        raise RuntimeError(f"No rows sampled for {model_name}")

    return pd.concat(sampled_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Build validation frame
# ---------------------------------------------------------------------------
def build_validation_frame(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the 200-row DataFrame for Excel export."""
    all_parts: List[pd.DataFrame] = []
    for model_name, df in dfs.items():
        sampled = sample_2x2_cells(df, model_name, N_PER_CELL, SEED)

        # Build per-row records
        records: List[Dict] = []
        for _, row in sampled.iterrows():
            spans = row.get("llm_hallucinated_spans", None)
            answer_marked = mark_spans_inline(
                str(row["generated_answer"]), spans
            )

            # Format raw spans as a readable string
            if spans is None or (isinstance(spans, float) and np.isnan(spans)):
                spans_str = ""
            elif isinstance(spans, (list, tuple, np.ndarray)):
                spans_str = " | ".join(str(s) for s in spans)
            else:
                spans_str = str(spans)

            records.append(
                {
                    "model": model_name,
                    "cell": row["cell"],
                    "question_id": int(row["question_id"]),
                    "evidence_present": int(row["evidence_present"]),
                    "question_sentence": str(row["question_sentence"]),
                    "evidence": str(row.get("evidence", "") or ""),
                    "generated_answer": str(row["generated_answer"]),
                    "generated_answer_marked": answer_marked,
                    "label_weak_hallucination": int(
                        row["label_weak_hallucination"]
                    ),
                    "label_weak_hallucination_score": float(
                        row["label_weak_hallucination_score"]
                    ),
                    "label_llm_answer": int(row["label_llm_answer"]),
                    "label_hallucination_confidence": float(
                        row["label_hallucination_confidence"]
                    ),
                    "llm_hallucinated_spans": spans_str,
                    # Empty columns for human annotation
                    "human_label": "",
                    "human_notes": "",
                }
            )

        all_parts.append(pd.DataFrame(records))

    result = pd.concat(all_parts, ignore_index=True)

    # Sort for readability: model -> cell -> question_id
    result = result.sort_values(
        ["model", "cell", "question_id"]
    ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Excel writing
# ---------------------------------------------------------------------------
def write_excel(df: pd.DataFrame, path: Path) -> None:
    """Write *df* to *path* as a formatted Excel spreadsheet."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="validation", index=False)

        workbook = writer.book
        worksheet = writer.sheets["validation"]

        # Formats
        header_fmt = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#D9E1F2",
                "border": 1,
                "text_wrap": True,
                "valign": "top",
            }
        )
        text_fmt = workbook.add_format(
            {"text_wrap": True, "valign": "top"}
        )
        text_fmt_no_wrap = workbook.add_format(
            {"valign": "top"}
        )
        annot_fmt = workbook.add_format(
            {"bg_color": "#FFF2CC", "valign": "top"}
        )

        # Column specifications: (col_idx, header, width, wrap?, annot?)
        col_specs = [
            (0, "model", 14, False, False),
            (1, "cell", 38, False, False),
            (2, "question_id", 14, False, False),
            (3, "evidence_present", 10, False, False),
            (4, "question_sentence", 50, True, False),
            (5, "evidence", 50, True, False),
            (6, "generated_answer", 60, True, False),
            (7, "generated_answer_marked", 60, True, False),
            (8, "label_weak_hallucination", 12, False, False),
            (9, "label_weak_hallucination_score", 12, False, False),
            (10, "label_llm_answer", 10, False, False),
            (11, "label_hallucination_confidence", 12, False, False),
            (12, "llm_hallucinated_spans", 40, True, False),
            (13, "human_label", 12, False, True),
            (14, "human_notes", 40, True, True),
        ]

        # Apply column formats and widths
        n_rows = len(df)
        for col_idx, header, width, wrap, annot in col_specs:
            if annot:
                fmt = annot_fmt
            elif wrap:
                fmt = text_fmt
            else:
                fmt = text_fmt_no_wrap
            worksheet.set_column(col_idx, col_idx, width, fmt)

        # Re-write header row with header format (set_column applies to data
        # cells only when used after to_excel, so we overwrite header manually)
        for col_idx, header, _, _, _ in col_specs:
            worksheet.write(0, col_idx, header, header_fmt)

        # Freeze top row and enable autofilter
        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, n_rows, len(col_specs) - 1)

    print(f"  Wrote {n_rows} rows to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading labelled data...")
    dfs = load_labeled_data()

    print("\nSampling 2x2 contingency cells...")
    val_df = build_validation_frame(dfs)

    print(f"\nTotal samples: {len(val_df)}")
    print("Breakdown:")
    print(val_df.groupby(["model", "cell"]).size().to_string())

    output_path = ANALYSIS_DIR / "human_validation_sample.xlsx"
    print(f"\nWriting Excel spreadsheet...")
    write_excel(val_df, output_path)

    print(f"\nDone.  Output: {output_path}")
    print("Manual annotation columns: human_label (0/1), human_notes (free text)")


if __name__ == "__main__":
    main()
