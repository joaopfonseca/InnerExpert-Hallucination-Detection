"""
7.0 Train-data analysis: export LaTeX examples, confidence tables, and prompt
templates for the labelled RealtimeQA training data.

Produces three fragment (.tex) files in the ``analysis/`` directory:

- ``examples.tex``        5 questions x 4 conditions (OLMoE/Gemma x base/evidence)
                           with hallucinated spans highlighted in red.
- ``confidence_table.tex``  Label-distribution and confidence tables.
- ``prompt_template.tex``  Generation and judge prompt templates.

Usage::

    python experiments/7.0-train-data-analysis.py

Paths to the labelled parquet files are hardcoded relative to the project
root.
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
# Paths (hardcoded, relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "realtimeqa-2024-2025"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

OLMOE_PARQUET = (
    DATA_DIR
    / "allenai__OLMoE-1B-7B-0924-Instruct"
    / "results_labeled_zai-org__GLM-5.1.parquet"
)
GEMMA_PARQUET = (
    DATA_DIR
    / "google__gemma-4-26B-A4B-it"
    / "results_labeled_zai-org__GLM-5.1.parquet"
)

# Curated example question IDs
EXAMPLE_QIDS = [
    202401058,   # Greta Gerwig - correct terse answer labeled hallucinated
    2024053125,   # Rehoboth Beach restaurant - clean refusal vs hedged fabrication
    2024010515,   # Covid vaccine percentage - refusal + fabrication
    202401052,   # Luke Humphries - correct with evidence vs verbose hallucination
    2025040416,   # Queen Mary 2 norovirus - evidence helps Gemma, not OLMoE
]

MODEL_LABELS = {
    "olmoe": "OLMoE-1B-7B",
    "gemma": "Gemma-4-26B",
}


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------
_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def escape_latex(text: str, escape_braces: bool = True) -> str:
    """Escape LaTeX special characters in *text*.

    When *escape_braces* is ``False``, curly braces are left as-is
    (used for prompt templates where ``{placeholder}`` syntax should
    be visible).
    """
    # Replace backslash FIRST so that backslashes introduced by later
    # replacements (e.g. \_ , \%) are not double-escaped.
    if "\\" in text:
        text = text.replace("\\", r"\textbackslash{}")
    for char, repl in _LATEX_SPECIAL.items():
        if char == "\\":
            continue
        if not escape_braces and char in "{}":
            continue
        text = text.replace(char, repl)
    return text


def highlight_spans(answer: str, spans: Optional[np.ndarray]) -> str:
    """Return LaTeX-escaped *answer* with each span wrapped in
    ``\\textcolor{red}{...}``.

    Spans are treated as exact substrings.  Overlapping spans are resolved by
    preferring the longer match.  Spans not found in *answer* are silently
    skipped.
    """
    if spans is None or len(spans) == 0:
        return escape_latex(answer)

    span_list: List[str] = [str(s) for s in spans if str(s).strip()]
    if not span_list:
        return escape_latex(answer)

    # Find all non-overlapping match intervals, preferring longer spans.
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
        return escape_latex(answer)

    # Sort by start, then by -length (longer first)
    intervals.sort(key=lambda iv: (iv[0], -(iv[1] - iv[0])))

    # Resolve overlaps: greedily pick non-overlapping intervals
    merged: List[Tuple[int, int]] = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            # Overlap: extend the previous interval if this one is longer
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # Build output
    parts: List[str] = []
    cursor = 0
    for start, end in merged:
        # Escape the text before the span
        if start > cursor:
            parts.append(escape_latex(answer[cursor:start]))
        # Wrap the span text in \textcolor{red}{...}
        span_text = escape_latex(answer[start:end])
        parts.append(r"\textcolor{red}{" + span_text + "}")
        cursor = end
    # Trailing text
    if cursor < len(answer):
        parts.append(escape_latex(answer[cursor:]))

    return "".join(parts)


def truncate(text: str, max_len: int = 400) -> str:
    """Truncate *text* to *max_len* characters, adding an ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + r" \ldots"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_labeled_data() -> Dict[str, pd.DataFrame]:
    """Load both models' labelled parquet files."""
    dfs = {}
    for name, path in [("olmoe", OLMOE_PARQUET), ("gemma", GEMMA_PARQUET)]:
        if not path.exists():
            raise FileNotFoundError(f"Labelled parquet not found: {path}")
        df = pd.read_parquet(path)
        df["evidence_present"] = df["evidence_present"].astype(bool)
        df["label_llm_answer"] = df["label_llm_answer"].astype(float)
        dfs[name] = df
        print(f"  {name}: {len(df)} rows from {path.name}")
    return dfs


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------
def get_question_rows(
    df: pd.DataFrame, qid: int
) -> Dict[bool, pd.Series]:
    """Return {evidence_present: row} for *qid*."""
    rows = {}
    sub = df[df["question_id"] == qid]
    for _, row in sub.iterrows():
        rows[row["evidence_present"]] = row
    return rows


def get_question_text(dfs: Dict[str, pd.DataFrame], qid: int) -> str:
    """Get the question text for *qid* from any available model."""
    for df in dfs.values():
        sub = df[df["question_id"] == qid]
        if len(sub) > 0:
            return sub.iloc[0]["question_sentence"]
    return ""


# ---------------------------------------------------------------------------
# examples.tex
# ---------------------------------------------------------------------------
def generate_examples_tex(dfs: Dict[str, pd.DataFrame]) -> str:
    """Generate the examples.tex fragment."""
    lines: List[str] = []
    lines.append(r"% Auto-generated by experiments/7.0-train-data-analysis.py")
    lines.append(r"% Requires: tcolorbox, alltt, xcolor")
    lines.append("")

    for idx, qid in enumerate(EXAMPLE_QIDS, 1):
        question = get_question_text(dfs, qid)
        olmoe_rows = get_question_rows(dfs["olmoe"], qid)
        gemma_rows = get_question_rows(dfs["gemma"], qid)

        lines.append(r"\begin{tcolorbox}[")
        lines.append(r"  colback=gray!8, colframe=black!60,")
        lines.append(r"  title={\textbf{Example %d} \small (qid=%d)}," % (idx, qid))
        lines.append(r"]")
        lines.append(r"\small")
        lines.append(r"\textbf{Question:} %s" % escape_latex(question))
        lines.append("")

        # Evidence (shared across both models for the same qid)
        ev_row = next(
            (r for r in list(olmoe_rows.values()) + list(gemma_rows.values())
             if r.get("evidence") and str(r["evidence"]).strip()),
            None,
        )
        if ev_row is not None:
            lines.append(r"\textbf{Evidence:} %s" % escape_latex(truncate(str(ev_row["evidence"]), 400)))
            lines.append("")

        for model_key, model_label in MODEL_LABELS.items():
            rows = olmoe_rows if model_key == "olmoe" else gemma_rows
            for ev_present in [False, True]:
                ev_label = "with evidence" if ev_present else "no evidence"
                row = rows.get(ev_present)
                if row is None:
                    lines.append(
                        r"\textbf{%s (%s)} \textit{(missing)}"
                        % (model_label, ev_label)
                    )
                    lines.append("")
                    continue

                label = int(row["label_llm_answer"])
                conf = row["label_hallucination_confidence"]
                label_str = "hallucinated" if label == 1 else "grounded"
                answer = row["generated_answer"]
                spans = row["llm_hallucinated_spans"]

                lines.append(
                    r"\textbf{%s (%s)} \hfill "
                    r"\textsc{label=%d, conf=%.3f (%s)}"
                    % (model_label, ev_label, label, conf, label_str)
                )
                lines.append(r"\begin{alltt}\small")
                highlighted = highlight_spans(answer, spans)
                lines.append(truncate(highlighted, 500))
                lines.append(r"\end{alltt}")
                lines.append("")

        lines.append(r"\end{tcolorbox}")
        lines.append("")
        lines.append(r"\vspace{1em}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table generators (each writes one standalone table fragment)
# ---------------------------------------------------------------------------
def generate_label_distribution_tex(dfs: Dict[str, pd.DataFrame]) -> str:
    """Table 1: label distribution by model and evidence condition."""
    lines: List[str] = []
    lines.append(r"% Auto-generated by experiments/7.0-train-data-analysis.py")
    lines.append(r"% Requires: booktabs, xcolor")
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Label distribution by model and evidence condition}")
    lines.append(r"\label{tab:label-distribution}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Condition & N & Halluc. & Grounded & Halluc.\,\% \\")
    lines.append(r"\midrule")

    for model_key, model_label in MODEL_LABELS.items():
        df = dfs[model_key]
        for ev_present in [False, True]:
            sub = df[df["evidence_present"] == ev_present]
            n = len(sub)
            n_hall = int(sub["label_llm_answer"].sum())
            n_grounded = n - n_hall
            pct = 100.0 * n_hall / n if n > 0 else 0.0
            cond = "Evidence" if ev_present else "Base"
            lines.append(
                r"%s & %s & %d & %d & %d & %.1f\,\%% \\"
                % (model_label, cond, n, n_hall, n_grounded, pct)
            )
        n = len(df)
        n_hall = int(df["label_llm_answer"].sum())
        n_grounded = n - n_hall
        pct = 100.0 * n_hall / n if n > 0 else 0.0
        lines.append(
            r"\textbf{%s} & \textbf{Overall} & \textbf{%d} & \textbf{%d} & \textbf{%d} & \textbf{%.1f\,\%%} \\"
            % (model_label, n, n_hall, n_grounded, pct)
        )
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_confidence_distribution_tex(dfs: Dict[str, pd.DataFrame]) -> str:
    """Table 2: confidence distribution for hallucinated labels."""
    lines: List[str] = []
    lines.append(r"% Auto-generated by experiments/7.0-train-data-analysis.py")
    lines.append(r"% Requires: booktabs, xcolor")
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Confidence distribution for hallucinated labels (label=1)}")
    lines.append(r"\label{tab:confidence-distribution}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Confidence & OLMoE count & OLMoE \% & Gemma count & Gemma \% \\")
    lines.append(r"\midrule")

    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    bin_labels = [
        r"%.1f $\leq$ conf $<$ %.1f" % (lo, hi) for lo, hi in bins
    ]
    for (lo, hi), blabel in zip(bins, bin_labels):
        counts = {}
        for model_key in ["olmoe", "gemma"]:
            df = dfs[model_key]
            hall = df[df["label_llm_answer"] == 1]
            total_hall = len(hall)
            n = int(((hall["label_hallucination_confidence"] >= lo) & (hall["label_hallucination_confidence"] < hi)).sum())
            pct = 100.0 * n / total_hall if total_hall > 0 else 0.0
            counts[model_key] = (n, pct, total_hall)

        lines.append(
            r"%s & %d & %.1f\,\%% & %d & %.1f\,\%% \\"
            % (
                blabel,
                counts["olmoe"][0],
                counts["olmoe"][1],
                counts["gemma"][0],
                counts["gemma"][1],
            )
        )

    lines.append(r"\midrule")
    totals = {}
    for model_key in ["olmoe", "gemma"]:
        df = dfs[model_key]
        totals[model_key] = int((df["label_llm_answer"] == 1).sum())
    lines.append(
        r"\textbf{Total} & \textbf{%d} & \textbf{100\,\%%} & \textbf{%d} & \textbf{100\,\%%} \\"
        % (totals["olmoe"], totals["gemma"])
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_label_asymmetry_tex(dfs: Dict[str, pd.DataFrame]) -> str:
    """Table 3: label distribution asymmetry summary."""
    lines: List[str] = []
    lines.append(r"% Auto-generated by experiments/7.0-train-data-analysis.py")
    lines.append(r"% Requires: booktabs, xcolor")
    lines.append("")

    metrics = []
    for model_key in ["olmoe", "gemma"]:
        df = dfs[model_key]
        total = len(df)
        hall = int((df["label_llm_answer"] == 1).sum())
        grounded = total - hall
        hall_pct = 100.0 * hall / total if total > 0 else 0.0
        ev_total = int((df["evidence_present"] == True).sum())
        base_total = int((df["evidence_present"] == False).sum())
        hall_ev = int(((df["label_llm_answer"] == 1) & (df["evidence_present"] == True)).sum())
        hall_base = int(((df["label_llm_answer"] == 1) & (df["evidence_present"] == False)).sum())
        hall_ev_pct = 100.0 * hall_ev / ev_total if ev_total > 0 else 0.0
        hall_base_pct = 100.0 * hall_base / base_total if base_total > 0 else 0.0
        grounded_ev = ev_total - hall_ev
        grounded_base = base_total - hall_base
        metrics.append({
            "total": total,
            "hall": hall,
            "grounded": grounded,
            "hall_pct": hall_pct,
            "ev_total": ev_total,
            "base_total": base_total,
            "hall_ev": hall_ev,
            "hall_base": hall_base,
            "hall_ev_pct": hall_ev_pct,
            "hall_base_pct": hall_base_pct,
            "grounded_ev": grounded_ev,
            "grounded_base": grounded_base,
        })

    row_defs = [
        ("Total samples", "total"),
        ("Hallucinated", "hall"),
        ("Grounded", "grounded"),
        ("Hallucinated (\\%)", "hall_pct", True),
        ("\\quad Base (no evidence)", "hall_base"),
        ("\\quad With evidence", "hall_ev"),
        ("\\quad Base halluc. (\\%)", "hall_base_pct", True),
        ("\\quad Evidence halluc. (\\%)", "hall_ev_pct", True),
        ("Grounded (base)", "grounded_base"),
        ("Grounded (evidence)", "grounded_ev"),
    ]

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Label distribution asymmetry summary}")
    lines.append(r"\label{tab:asymmetry}")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Metric & OLMoE & Gemma \\")
    lines.append(r"\midrule")

    for rd in row_defs:
        label = rd[0]
        key = rd[1]
        is_pct = len(rd) == 3 and rd[2]
        v_olmoe = metrics[0][key]
        v_gemma = metrics[1][key]
        if is_pct:
            lines.append(
                r"%s & %.1f\,\%% & %.1f\,\%% \\" % (label, v_olmoe, v_gemma)
            )
        else:
            lines.append(r"%s & %d & %d \\" % (label, v_olmoe, v_gemma))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# prompt_template.tex
# ---------------------------------------------------------------------------
GENERATION_SYSTEM_PROMPT = (
    "You are a helpful assistant who provides accurate and "
    "very concise answers to questions about recent "
    "events. Today is {date}."
)
GENERATION_USER_PROMPT_BASE = "{question}"
GENERATION_USER_PROMPT_EVIDENCE = "Evidence: {evidence}\n\nQuestion: {question}"

JUDGE_PROMPT_TEMPLATE = (
    "Given a question, evidence, and model answer, decide whether the answer "
    "contains unsupported or hallucinated content.\n"
    'Return valid JSON only with this exact schema:\n'
    '{{"label": 1 or 0, '
    '"hallucinated_spans": ["exact substring from answer", ...]}}\n'
    "Rules:\n"
    "- label = 1 means hallucinated/unsupported, label = 0 means grounded/correct.\n"
    "- hallucinated_spans must be exact substrings from the answer.\n"
    "- if label = 0, hallucinated_spans must be an empty list.\n\n"
    "Question: {question}\n"
    "Evidence: {evidence}\n"
    "Answer: {answer}\n"
)
JUDGE_SYSTEM_PROMPT = "Return concise, valid JSON only."


def generate_prompt_template_tex() -> str:
    """Generate the prompt_template.tex fragment."""
    lines: List[str] = []
    lines.append(r"% Auto-generated by experiments/7.0-train-data-analysis.py")
    lines.append(r"% Requires: tcolorbox, alltt")
    lines.append("")

    # --- Generation prompt ---
    lines.append(r"\begin{tcolorbox}[")
    lines.append(r"  colback=blue!5, colframe=blue!50,")
    lines.append(r"  title={\textbf{Generation Prompt (tokenize\_realtimeqa)}},")
    lines.append(r"]")
    lines.append(r"\small")
    lines.append(r"\textbf{System message:}")
    lines.append(r"\begin{alltt}\small")
    lines.append(escape_latex(GENERATION_SYSTEM_PROMPT, escape_braces=False))
    lines.append(r"\end{alltt}")
    lines.append("")
    lines.append(r"\textbf{User message (base, no evidence):}")
    lines.append(r"\begin{alltt}\small")
    lines.append(escape_latex(GENERATION_USER_PROMPT_BASE, escape_braces=False))
    lines.append(r"\end{alltt}")
    lines.append("")
    lines.append(r"\textbf{User message (with evidence):}")
    lines.append(r"\begin{alltt}\small")
    lines.append(escape_latex(GENERATION_USER_PROMPT_EVIDENCE, escape_braces=False))
    lines.append(r"\end{alltt}")
    lines.append(r"\end{tcolorbox}")
    lines.append("")
    lines.append(r"\vspace{1em}")
    lines.append("")

    # --- Judge prompt ---
    lines.append(r"\begin{tcolorbox}[")
    lines.append(r"  colback=red!5, colframe=red!50,")
    lines.append(r"  title={\textbf{Judge Prompt (2.0-make-labels.py)}},")
    lines.append(r"]")
    lines.append(r"\small")
    lines.append(r"\textbf{System message:}")
    lines.append(r"\begin{alltt}\small")
    lines.append(escape_latex(JUDGE_SYSTEM_PROMPT, escape_braces=False))
    lines.append(r"\end{alltt}")
    lines.append("")
    lines.append(r"\textbf{User message template:}")
    lines.append(r"\begin{alltt}\small")
    lines.append(escape_latex(JUDGE_PROMPT_TEMPLATE, escape_braces=False))
    lines.append(r"\end{alltt}")
    lines.append(r"\end{tcolorbox}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading labelled data...")
    dfs = load_labeled_data()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating examples.tex...")
    examples_tex = generate_examples_tex(dfs)
    (ANALYSIS_DIR / "examples.tex").write_text(examples_tex, encoding="utf-8")
    print(f"  -> {ANALYSIS_DIR / 'examples.tex'}")

    print("Generating table fragments...")
    for name, gen_fn in [
        ("label_distribution.tex", generate_label_distribution_tex),
        ("confidence_distribution.tex", generate_confidence_distribution_tex),
        ("label_asymmetry.tex", generate_label_asymmetry_tex),
    ]:
        tex = gen_fn(dfs)
        path = ANALYSIS_DIR / name
        path.write_text(tex, encoding="utf-8")
        print(f"  -> {path}")

    print("Generating prompt_template.tex...")
    prompt_tex = generate_prompt_template_tex()
    (ANALYSIS_DIR / "prompt_template.tex").write_text(prompt_tex, encoding="utf-8")
    print(f"  -> {ANALYSIS_DIR / 'prompt_template.tex'}")

    print("\nDone.  Outputs written to analysis/:")

    # Print quick summary
    for model_key, model_label in MODEL_LABELS.items():
        df = dfs[model_key]
        n = len(df)
        hall = int((df["label_llm_answer"] == 1).sum())
        print(f"  {model_label}: {n} rows, {hall} hallucinated ({100*hall/n:.1f}%)")


if __name__ == "__main__":
    main()