"""6.1 — OOS hallucination labeling via LLM-as-judge (GLM-5.1).

Labels OOS dataset generations with the same LLM-as-judge approach as
``2.0-make-labels.py`` (GLM-5.1 via DeepInfra), but **without** the
weak-label step (base/RAG metric separability) since OOS datasets have
ground-truth answers, not base/RAG pairs.

Rows where the LLM judge fails or returns ``label=None`` are **dropped**
(no weak-label fallback).

Outputs:
  - data/<dataset_slug>/<model_slug>/results_labeled_{label_model_slug}.parquet

Columns:
  - All original columns from results.parquet (expanded to base/evidence rows)
  - evidence_present (0 or 1)
  - label_llm_answer (1 or 0; NaN for rows where LLM failed → dropped)
  - llm_hallucinated_spans (list of substring strings, or None)

Usage:
    python 6.1-oos-label.py --dataset squad --label-model zai-org/GLM-5.1
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

try:
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

from moeuncert.datasets_adapters import get_adapter, list_oos_datasets
from moeuncert.experiments import resolve_model_slug


# ---------------------------------------------------------------------------
# Import shared helpers from 2.0-make-labels.py (digit-prefixed filename
# is not a valid module name, so we load it via importlib).
# ---------------------------------------------------------------------------

def _load_make_labels_module():
    """Load experiments/2.0-make-labels.py as a module and return it."""
    module_path = Path(__file__).parent / "2.0-make-labels.py"
    spec = importlib.util.spec_from_file_location("_make_labels", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_make_labels = _load_make_labels_module()
expand_base_rag_rows = _make_labels.expand_base_rag_rows
PROMPT_TEMPLATE = _make_labels.PROMPT_TEMPLATE
generate_llm_labels_and_spans = _make_labels.generate_llm_labels_and_spans


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Label OOS dataset generations via LLM-as-judge (6.1)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list_oos_datasets(),
        help="OOS dataset name",
    )
    parser.add_argument(
        "--model", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="Subject model name (the model whose outputs are labeled)",
    )
    parser.add_argument(
        "--label-model", type=str, default="zai-org/GLM-5.1",
        help="LLM-as-judge model (via DeepInfra)",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root data directory",
    )
    parser.add_argument(
        "--input-file", type=str, default="results.parquet",
        help="Input results file (default: results.parquet)",
    )
    parser.add_argument(
        "--max-llm-samples", type=int, default=None,
        help="Max rows to label (default: all rows)",
    )

    args = parser.parse_args()

    # Load .env for DEEPINFRA_API_KEY
    from dotenv import load_dotenv
    load_dotenv()

    print(f"{'=' * 70}")
    print(f"6.1 — OOS LABELING ({args.dataset})")
    print(f"{'=' * 70}")

    adapter = get_adapter(args.dataset)
    model_slug = resolve_model_slug(args.model)
    model_dir = args.data_root / adapter.slug / model_slug

    results_path = model_dir / args.input_file
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}\n"
            f"Run 6.0-oos-generate.py --dataset {args.dataset} first."
        )

    print(f"\nLoading {results_path}...")
    df = pd.read_parquet(results_path)
    print(f"  {len(df)} rows")

    # --- Expand base + evidence rows --------------------------------------
    # expand_base_rag_rows splits _rag columns into separate rows with
    # evidence_present flag.  All OOS datasets now have evidence-conditioned
    # generations (using answer_str as evidence for datasets without native
    # context), so always expand.
    df_expanded = expand_base_rag_rows(df, has_evidence=True)
    print(f"  After base/evidence expansion: {len(df_expanded)} rows")

    # --- Fill empty evidence with reference answer -------------------------
    # For datasets without a context passage (TruthfulQA, NQ-Open, FreshQA),
    # the evidence column is empty.  The LLM judge needs a reference to
    # compare the generated answer against, so we use the ground-truth
    # answer_str as the evidence field.
    if not adapter.has_evidence:
        df_expanded["evidence"] = df_expanded["answer_str"]
        print(f"  Filled evidence with answer_str (non-evidence dataset)")
    else:
        empty_mask = df_expanded["evidence"].astype(str).str.len() == 0
        if empty_mask.any():
            df_expanded.loc[empty_mask, "evidence"] = df_expanded.loc[empty_mask, "answer_str"]
            print(f"  Filled {empty_mask.sum()} empty evidence rows with answer_str")

    # --- LLM-as-judge labeling --------------------------------------------
    label_model_slug = args.label_model.replace("/", "__")
    output_file = model_dir / f"results_labeled_{label_model_slug}.parquet"
    if output_file.exists():
        raise FileExistsError(
            f"Output already exists: {output_file}\n"
            f"Delete it to re-run labeling."
        )

    # Check API key
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        print("\nWARNING: DEEPINFRA_API_KEY not set. Skipping LLM labeling.")
        print("  Only weak labels will be produced (if applicable).")
        llm_labels = pd.Series(dtype=float, index=df_expanded.index)
        llm_spans: dict = {}
    else:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=api_key,
        )
        print(f"\nLabeling via DeepInfra (model={args.label_model})...")
        llm_labels, llm_spans = generate_llm_labels_and_spans(
            df_expanded, client, args.label_model,
            max_samples=args.max_llm_samples,
        )

        n_labeled = llm_labels.notna().sum()
        n_failed = llm_labels.isna().sum()
        print(f"  Labeled: {n_labeled}, Failed: {n_failed}")

    # --- Build output DataFrame -------------------------------------------
    df_expanded["label_llm_answer"] = llm_labels
    df_expanded["llm_hallucinated_spans"] = df_expanded.index.map(
        lambda idx: llm_spans.get(idx, None)
    )

    # --- Drop rows where LLM failed (no weak-label fallback for OOS) ------
    n_before = len(df_expanded)
    df_expanded = df_expanded[df_expanded["label_llm_answer"].notna()].copy()
    df_expanded["label_llm_answer"] = df_expanded["label_llm_answer"].astype(int)
    n_after = len(df_expanded)
    n_dropped = n_before - n_after
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows where LLM label failed (no weak-label fallback)")
    print(f"  Final labeled rows: {n_after}")

    # --- Save -------------------------------------------------------------
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_expanded.to_parquet(output_file, index=False)
    print(f"\n  Saved {output_file}")

    print(f"\n{'=' * 70}")
    print("LABELING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Next step: python 7.0-oos-evaluation.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()