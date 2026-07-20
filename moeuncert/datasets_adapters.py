"""Dataset adapters for out-of-sample (OOS) evaluation.

Each adapter normalises a different public dataset to a common schema so
that the OOS pipeline (6.0 / 6.1 / 7.0 / 8.0) can treat them uniformly.

Common schema (produced by ``to_common_schema``):
    - question_id      : str  — unique question identifier
    - question_sentence: str  — the question text (or task instruction)
    - evidence         : str  — evidence/context ("" if not available)
    - answer_str       : str  — ground-truth reference answer

Adapters also expose:
    - has_evidence     : bool — whether the dataset has an evidence field
    - date_str         : str  — fixed date for the system prompt
    - build_system_prompt() / build_user_prompt(row, with_evidence)
    - slug             : str  — directory name ("oos-squad", etc.)

Registry: ``OOS_ADAPTERS`` maps ``dataset_name -> adapter_class``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

import pandas as pd

from .datasets import (
    fetch_freshqa,
    fetch_nq_open,
    fetch_squad,
    fetch_truthfulqa,
)


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


class OOSDatasetAdapter:
    """Base class for OOS dataset adapters.

    Subclasses must set:
        name          — short identifier ("squad", "truthfulqa", ...)
        slug          — directory slug ("oos-squad", "oos-truthfulqa", ...)
        question_col  — column with the question text
        evidence_col  — column with evidence/context (or None)
        answer_col    — column (or columns) with ground-truth answer(s)
        id_col        — column with a unique question id (or None → generated)
        has_evidence  — bool
        date_str      — fixed date string for the system prompt
        task_type     — "qa" or "summarization"
    """

    name: str = ""
    slug: str = ""
    question_col: str = ""
    evidence_col: Optional[str] = None
    answer_col = None  # str or list[str]
    id_col: Optional[str] = None
    has_evidence: bool = False
    date_str: str = "January 1, 2024"
    task_type: str = "qa"

    # --- Dataset fetch ----------------------------------------------------

    def fetch(self) -> pd.DataFrame:
        """Fetch the raw dataset. Override in subclasses."""
        raise NotImplementedError

    # --- Schema normalisation ---------------------------------------------

    def to_common_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise the raw dataset to the common schema.

        Returns a DataFrame with columns:
            question_id, question_sentence, evidence, answer_str
        """
        out = pd.DataFrame()

        # question_id
        if self.id_col is not None and self.id_col in df.columns:
            out["question_id"] = df[self.id_col].astype(str)
        else:
            out["question_id"] = [f"{self.name}_{i}" for i in range(len(df))]

        # question_sentence
        out["question_sentence"] = df[self.question_col].astype(str)

        # evidence
        if self.evidence_col is not None and self.evidence_col in df.columns:
            out["evidence"] = df[self.evidence_col].astype(str).fillna("")
        else:
            out["evidence"] = ""

        # answer_str — may be a single column or a list of columns
        out["answer_str"] = self._extract_answers(df)

        return out

    def _extract_answers(self, df: pd.DataFrame) -> pd.Series:
        """Extract ground-truth answers as a single string per row."""
        if isinstance(self.answer_col, (list, tuple)):
            # Join multiple answer candidates with " | "
            parts = []
            for col in self.answer_col:
                if col in df.columns:
                    parts.append(df[col].astype(str))
            if not parts:
                return pd.Series([""] * len(df))
            return parts[0].str.cat(parts[1:], sep=" | ")
        elif isinstance(self.answer_col, str) and self.answer_col in df.columns:
            val = df[self.answer_col]
            # If the column holds lists (e.g. SQuAD answers.text), join them
            if val.dtype == object and val.apply(lambda x: isinstance(x, (list, tuple))).any():
                return val.apply(
                    lambda x: " | ".join(str(a) for a in x) if isinstance(x, (list, tuple)) else str(x)
                )
            return val.astype(str)
        else:
            return pd.Series([""] * len(df))

    # --- Prompt construction ----------------------------------------------

    def build_system_prompt(self) -> str:
        """System prompt for the chat template."""
        if self.task_type == "summarization":
            return (
                "You are a helpful assistant who provides accurate and"
                " concise summaries of documents. Today is "
                f"{self.date_str}."
            )
        return (
            "You are a helpful assistant who provides accurate and"
            " very concise answers to questions. Today is "
            f"{self.date_str}."
        )

    def build_user_prompt(self, row: pd.Series, with_evidence: bool) -> str:
        """User prompt for a single row.

        For QA datasets: ``"Evidence: {ev}\n\nQuestion: {q}"`` or just ``{q}``.
        For summarization: ``"Summarize the following document:\n\n{document}"``.
        """
        if self.task_type == "summarization":
            doc = row.get("evidence", "") or row.get("question_sentence", "")
            return f"Summarize the following document:\n\n{doc}"

        q = row["question_sentence"]
        if with_evidence and self.has_evidence:
            ev = row.get("evidence", "")
            if ev:
                return f"Evidence: {ev}\n\nQuestion: {q}"
        return q

    # --- Tokenisation helper ----------------------------------------------

    def tokenize(self, tokenizer, df: pd.DataFrame, with_evidence: bool = False):
        """Apply the chat template and tokenize the common-schema DataFrame.

        Returns the tokenizer output (return_tensors="pt" by default,
        matching ``tokenize_realtimeqa``'s contract).
        """
        texts = []
        for _, row in df.iterrows():
            messages = [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": self.build_user_prompt(row, with_evidence)},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            texts.append(text)
        return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------


class SQuADAdapter(OOSDatasetAdapter):
    name = "squad"
    slug = "oos-squad"
    question_col = "question"
    evidence_col = "context"
    answer_col = "answers"
    id_col = "id"
    has_evidence = True
    task_type = "qa"

    def fetch(self) -> pd.DataFrame:
        return fetch_squad(split="validation")

    def _extract_answers(self, df: pd.DataFrame) -> pd.Series:
        # SQuAD `answers` column is a dict with `text` (list of strings)
        if "answers" in df.columns:
            return df["answers"].apply(
                lambda a: " | ".join(a.get("text", []))
                if isinstance(a, dict) and isinstance(a.get("text"), list)
                else (str(a) if a else "")
            )
        return pd.Series([""] * len(df))


class TruthfulQAAdapter(OOSDatasetAdapter):
    name = "truthfulqa"
    slug = "oos-truthfulqa"
    question_col = "Question"
    evidence_col = None
    answer_col = "Best Answer"
    id_col = None
    has_evidence = False
    task_type = "qa"

    def fetch(self) -> pd.DataFrame:
        return fetch_truthfulqa()


class NQOpenAdapter(OOSDatasetAdapter):
    name = "nq_open"
    slug = "oos-nq_open"
    question_col = "question"
    evidence_col = None
    answer_col = "answer"
    id_col = None
    has_evidence = False
    task_type = "qa"

    def fetch(self) -> pd.DataFrame:
        return fetch_nq_open(split="validation")

    def _extract_answers(self, df: pd.DataFrame) -> pd.Series:
        if "answer" in df.columns:
            return df["answer"].apply(
                lambda a: " | ".join(str(x) for x in a)
                if isinstance(a, (list, tuple)) else str(a)
            )
        return pd.Series([""] * len(df))


class FreshQAAdapter(OOSDatasetAdapter):
    name = "freshqa"
    slug = "oos-freshqa"
    question_col = "question"
    evidence_col = None  # may be set at runtime if column exists
    answer_col = "answer_0"  # primary answer; remaining answers are alternatives
    id_col = None
    has_evidence = False  # set dynamically in fetch
    task_type = "qa"

    def fetch(self) -> pd.DataFrame:
        df = fetch_freshqa()
        # FreshQA may have evidence-like columns; check at runtime.
        for candidate in ("more_info", "evidence_sources", "evidence"):
            if candidate in df.columns:
                self.evidence_col = candidate
                self.has_evidence = True
                break
        return df


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


OOS_ADAPTERS: Dict[str, Type[OOSDatasetAdapter]] = {
    "squad": SQuADAdapter,
    "truthfulqa": TruthfulQAAdapter,
    "nq_open": NQOpenAdapter,
    "freshqa": FreshQAAdapter,
}


def get_adapter(name: str) -> OOSDatasetAdapter:
    """Instantiate an adapter by dataset name."""
    if name not in OOS_ADAPTERS:
        raise ValueError(
            f"Unknown OOS dataset '{name}'. "
            f"Available: {list(OOS_ADAPTERS.keys())}"
        )
    return OOS_ADAPTERS[name]()


def list_oos_datasets() -> List[str]:
    """Return the list of available OOS dataset names."""
    return list(OOS_ADAPTERS.keys())