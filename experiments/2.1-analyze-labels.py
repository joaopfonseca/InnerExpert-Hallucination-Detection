"""
Analyze labeled RealtimeQA outputs to find interesting patterns.

This script helps identify specific cases such as:
- Hallucinations despite evidence being provided
- Correct answers without evidence
- High confidence hallucinations
- Edge cases for further investigation
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

from moeuncert.experiments import resolve_model_slug, resolve_dataset_slug


def analyze_hallucination_patterns(df):
    """Analyze and print hallucination patterns in the dataset."""
    
    print("=" * 80)
    print("HALLUCINATION PATTERN ANALYSIS")
    print("=" * 80)
    
    # Overall statistics
    print("\n### Overall Statistics ###")
    print(f"Total samples: {len(df)}")
    print(f"With evidence: {df['evidence_present'].sum()} ({df['evidence_present'].sum()/len(df)*100:.1f}%)")
    print(f"Without evidence: {(~df['evidence_present'].astype(bool)).sum()}")
    print(f"Hallucinated (LLM label): {df['label_llm_answer'].sum():.0f} ({df['label_llm_answer'].sum()/len(df)*100:.1f}%)")
    print(f"Not hallucinated: {(~df['label_llm_answer'].astype(bool)).sum()}")
    
    # Cross-tabulation
    print("\n### Evidence vs Hallucination Cross-tabulation ###")
    crosstab = pd.crosstab(
        df['evidence_present'],
        df['label_llm_answer'],
        rownames=['Evidence Present'],
        colnames=['Hallucinated (1=Yes, 0=No)'],
        margins=True
    )
    print(crosstab)
    
    # Calculate rates
    print("\n### Hallucination Rates by Evidence Condition ###")
    with_evidence = df[df['evidence_present'] == 1]
    without_evidence = df[df['evidence_present'] == 0]
    
    if len(with_evidence) > 0:
        hall_rate_with = with_evidence['label_llm_answer'].mean()
        print(f"WITH evidence: {hall_rate_with*100:.1f}% hallucinated ({with_evidence['label_llm_answer'].sum():.0f}/{len(with_evidence)})")
    
    if len(without_evidence) > 0:
        hall_rate_without = without_evidence['label_llm_answer'].mean()
        print(f"WITHOUT evidence: {hall_rate_without*100:.1f}% hallucinated ({without_evidence['label_llm_answer'].sum():.0f}/{len(without_evidence)})")


def find_hallucinations_with_evidence(df, n=10, save_path=None):
    """Find cases where model hallucinated despite evidence being provided."""
    
    print("\n" + "=" * 80)
    print("CASES: HALLUCINATED DESPITE EVIDENCE PROVIDED")
    print("=" * 80)
    
    # Filter: evidence_present=1 AND label_llm_answer=1 (hallucinated)
    cases = df[(df['evidence_present'] == 1) & (df['label_llm_answer'] == 1)].copy()
    
    print(f"\nFound {len(cases)} cases where evidence was provided but model hallucinated.")
    
    if len(cases) == 0:
        print("No cases found!")
        return None
    
    # Sort by hallucination confidence (highest first)
    if 'label_hallucination_confidence' in cases.columns:
        cases = cases.sort_values('label_hallucination_confidence', ascending=False)
    
    # Display first n cases
    print(f"\nShowing top {min(n, len(cases))} cases:\n")
    
    for i, (idx, row) in enumerate(cases.head(n).iterrows(), 1):
        print(f"\n{'─' * 80}")
        print(f"CASE {i} (Index: {idx})")
        print(f"{'─' * 80}")
        print(f"Question: {row['question_sentence']}")
        print(f"\nCorrect Answer: {row['answer_str']}")
        print(f"\nEvidence Provided:\n{row['evidence'][:500]}{'...' if len(str(row['evidence'])) > 500 else ''}")
        print(f"\nGenerated Answer: {row['generated_answer']}")
        
        if 'llm_hallucinated_spans' in row.index:
            spans = row['llm_hallucinated_spans']
            if spans is not None and (not isinstance(spans, float) or not pd.isna(spans)):
                print(f"\nIdentified Hallucinated Spans: {spans}")
        
        if 'label_hallucination_confidence' in row:
            print(f"\nHallucination Confidence: {row['label_hallucination_confidence']:.3f}")
        
        # Show similarity metrics if available
        metrics_to_show = ['rougeL', 'bert_f1', 'bleu']
        metrics_str = []
        for metric in metrics_to_show:
            if metric in row and pd.notna(row[metric]):
                metrics_str.append(f"{metric}: {row[metric]:.3f}")
        if metrics_str:
            print(f"Metrics: {', '.join(metrics_str)}")
    
    # Save to file if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Select relevant columns
        cols_to_save = [
            'question_id', 'question_sentence', 'answer_str', 'evidence',
            'generated_answer', 'label_llm_answer', 'label_hallucination_confidence',
            'rougeL', 'bert_f1', 'bleu'
        ]
        cols_to_save = [c for c in cols_to_save if c in cases.columns]
        
        cases[cols_to_save].to_csv(save_path, index=False)
        print(f"\n✅ Saved {len(cases)} cases to: {save_path}")
    
    return cases


def find_correct_without_evidence(df, n=10):
    """Find cases where model answered correctly without evidence."""
    
    print("\n" + "=" * 80)
    print("CASES: CORRECT ANSWERS WITHOUT EVIDENCE")
    print("=" * 80)
    
    # Filter: evidence_present=0 AND label_llm_answer=0 (not hallucinated)
    cases = df[(df['evidence_present'] == 0) & (df['label_llm_answer'] == 0)].copy()
    
    print(f"\nFound {len(cases)} cases where model answered correctly without evidence.")
    
    if len(cases) == 0:
        print("No cases found!")
        return None
    
    # Sort by confidence
    if 'label_hallucination_confidence' in cases.columns:
        cases = cases.sort_values('label_hallucination_confidence', ascending=True)
    
    print(f"\nShowing top {min(n, len(cases))} cases:\n")
    
    for i, (idx, row) in enumerate(cases.head(n).iterrows(), 1):
        print(f"\n{'─' * 80}")
        print(f"CASE {i} (Index: {idx})")
        print(f"{'─' * 80}")
        print(f"Question: {row['question_sentence']}")
        print(f"Correct Answer: {row['answer_str']}")
        print(f"Generated Answer: {row['generated_answer']}")
        
        if 'label_hallucination_confidence' in row:
            print(f"Hallucination Confidence: {row['label_hallucination_confidence']:.3f}")
    
    return cases


def find_high_confidence_hallucinations(df, threshold=0.8, n=10):
    """Find high-confidence hallucinations (model was very wrong but confident)."""
    
    print("\n" + "=" * 80)
    print(f"CASES: HIGH CONFIDENCE HALLUCINATIONS (confidence > {threshold})")
    print("=" * 80)
    
    if 'label_hallucination_confidence' not in df.columns:
        print("Hallucination confidence scores not available.")
        return None
    
    # Filter: hallucinated AND high confidence
    cases = df[
        (df['label_llm_answer'] == 1) & 
        (df['label_hallucination_confidence'] > threshold)
    ].copy()
    
    print(f"\nFound {len(cases)} high-confidence hallucinations.")
    
    if len(cases) == 0:
        print("No cases found!")
        return None
    
    cases = cases.sort_values('label_hallucination_confidence', ascending=False)
    
    print(f"\nShowing top {min(n, len(cases))} cases:\n")
    
    for i, (idx, row) in enumerate(cases.head(n).iterrows(), 1):
        print(f"\n{'─' * 80}")
        print(f"CASE {i} (Index: {idx})")
        print(f"{'─' * 80}")
        print(f"Question: {row['question_sentence']}")
        print(f"Evidence Present: {'Yes' if row['evidence_present'] else 'No'}")
        print(f"Correct Answer: {row['answer_str']}")
        print(f"Generated Answer: {row['generated_answer']}")
        print(f"Hallucination Confidence: {row['label_hallucination_confidence']:.3f}")
    
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze labeled RealtimeQA outputs for interesting patterns."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="HuggingFace model name (default: allenai/OLMoE-1B-7B-0924-Instruct)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Year(s) for RealtimeQA, e.g. --years 2025 2026. Defaults to 2026.",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        choices=range(1, 13),
        metavar="MONTH",
        help="Month (1-12). Only valid when a single year is provided.",
    )
    parser.add_argument(
        "--label-model",
        type=str,
        default="zai-org/GLM-5.1",
        help="Label model identifier (default: zai-org/GLM-5.1)",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        choices=["all", "hallucinate-with-evidence", "correct-without-evidence", "high-confidence"],
        default="all",
        help="Type of analysis to run (default: all)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of examples to display per category (default: 10)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save hallucinations-with-evidence cases to CSV",
    )
    args = parser.parse_args()
    
    # Resolve paths
    model_slug = resolve_model_slug(args.model)
    years, month, dataset_slug = resolve_dataset_slug(args.years, args.month)
    
    # Load labeled data
    data_dir = Path("data") / dataset_slug / model_slug
    labeled_file = data_dir / f"results_labeled_{args.label_model}.parquet"
    
    print(f"Loading labeled data from: {labeled_file}")
    df = pd.read_parquet(labeled_file)
    print(f"Loaded {len(df)} samples.\n")
    
    # Run analyses
    if args.analysis in ["all", "patterns"]:
        analyze_hallucination_patterns(df)
    
    if args.analysis in ["all", "hallucinate-with-evidence"]:
        save_path = data_dir / "hallucinations_with_evidence.csv" if args.save else None
        find_hallucinations_with_evidence(df, n=args.n, save_path=save_path)
    
    if args.analysis in ["all", "correct-without-evidence"]:
        find_correct_without_evidence(df, n=args.n)
    
    if args.analysis in ["all", "high-confidence"]:
        find_high_confidence_hallucinations(df, n=args.n)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
