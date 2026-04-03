"""Markdown Table QA environment for GRPO.

Single-turn: model receives a markdown table + question, produces
<think>reasoning</think> followed by a final answer.

Single reward signal: pandas-verified correctness. No format or
faithfulness rewards — keep the gradient signal sharp for small models.
"""
from __future__ import annotations

import re

import pandas as pd
import verifiers as vf
from datasets import Dataset as HFDataset, load_dataset

# ── Constants ────────────────────────────────────────────────────────────────

MATH_TYPES = {"sum", "mean", "percentage", "filtered_sum", "filtered_count"}
ENTITY_TYPES = {"max_row", "min_row", "rank_top3"}

SYSTEM_PROMPT = (
    "You are a markdown table expert. Given a markdown table and a question, "
    "reason step-by-step inside <think>...</think> tags, then give your final "
    "answer on a new line after the closing tag.\n\n"
    "Rules:\n"
    "- Always quote exact cell values from the table\n"
    "- Show arithmetic step by step: a + b = X; X + c = Y\n"
    "- Count only data rows (not the header). List them by name.\n"
    "- Final answer must be outside the think tags, clean and concise"
)


# ── Table parsing ────────────────────────────────────────────────────────────

def _parse_markdown_table(md: str) -> pd.DataFrame | None:
    lines = [l.strip() for l in md.strip().split("\n") if l.strip()]
    table_lines = [l for l in lines if l.startswith("|")]
    if len(table_lines) < 3:
        return None
    header = table_lines[0]
    data_lines = [l for l in table_lines[2:] if not re.match(r"^\|[\s\-|]+\|$", l)]
    cols = [c.strip() for c in header.split("|")[1:-1]]
    rows = []
    for line in data_lines:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) == len(cols):
            rows.append(cells)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=cols)


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.str.replace(r"[$€£¥%,]", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


# ── Extraction helpers ───────────────────────────────────────────────────────

def _strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    if "\n\n" in text:
        return text.split("\n\n")[-1].strip()
    return text.strip()


def _extract_numbers(text: str) -> list[float]:
    text = re.sub(r"[$€£¥%,]", "", text)
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _extract_final_number(text: str, qtype: str = "") -> float | None:
    text = re.sub(r"[$€£¥]", "", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    if qtype == "percentage":
        pct = re.findall(r"(-?\d+(?:\.\d+)?)\s*%", text)
        if pct:
            return float(pct[-1])
    nums = _extract_numbers(text)
    return nums[-1] if nums else None


def _approx_equal(a: float, b: float, rel_tol: float = 0.01, abs_tol: float = 0.5) -> bool:
    return abs(a - b) <= abs_tol or (b != 0 and abs(a - b) / abs(b) <= rel_tol)


def _get_text(completion) -> str:
    if isinstance(completion, list) and completion:
        return completion[-1]["content"]
    return str(completion)


def _token_f1(pred: str, ref: str) -> float:
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "of", "in", "on", "at", "to",
        "for", "with", "by", "from", "up", "about", "into", "and", "or", "but",
        "not", "so", "it", "its", "this", "that", "there", "here",
    }
    pred_toks = set(re.findall(r"\b[\w.]+\b", pred.lower())) - stopwords
    ref_toks = set(re.findall(r"\b[\w.]+\b", ref.lower())) - stopwords
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    common = pred_toks & ref_toks
    p = len(common) / len(pred_toks)
    r = len(common) / len(ref_toks)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── Gold computation via pandas ──────────────────────────────────────────────

def _find_column(df: pd.DataFrame, hint: str) -> str | None:
    hint_l = hint.lower()
    for col in df.columns:
        if hint_l in col.lower() or col.lower() in hint_l:
            return col
    return None


def _compute_gold(question: str, table_md: str, qtype: str) -> dict:
    df = _parse_markdown_table(table_md)
    if df is None:
        return {"value": None, "method": "failed"}

    q = question.lower()
    n_rows = len(df)

    try:
        if qtype == "sum":
            for col in df.columns:
                if col.lower() in q:
                    s = _to_numeric(df[col])
                    if s.notna().any():
                        return {"value": round(s.sum(), 2), "method": "pandas"}

        elif qtype == "mean":
            for col in df.columns:
                if col.lower() in q:
                    s = _to_numeric(df[col])
                    if s.notna().any():
                        return {"value": round(s.mean(), 2), "method": "pandas"}

        elif qtype == "percentage":
            m = re.search(r"have\s+(\w[\w\s]*?)\s+['\"](.+?)['\"]", question, re.IGNORECASE)
            if not m:
                m = re.search(r"have\s+(\w[\w\s]*?)\s+(.+?)[\?\.]?\s*$", question, re.IGNORECASE)
            if m:
                col_hint, val = m.group(1).strip(), m.group(2).strip().strip("'\"")
                col = _find_column(df, col_hint)
                if col:
                    count = int((df[col].str.strip() == val).sum())
                    pct = round(count / n_rows * 100, 1)
                    return {"value": pct, "method": "pandas"}

        elif qtype == "filtered_count":
            m = re.search(r"have\s+(\w[\w\s]*?)\s+['\"](.+?)['\"]", question, re.IGNORECASE)
            if not m:
                m = re.search(r"have\s+(\w[\w\s]*?)\s+(.+?)[\?\.]?\s*$", question, re.IGNORECASE)
            if m:
                col_hint, val = m.group(1).strip(), m.group(2).strip().strip("'\"")
                col = _find_column(df, col_hint)
                if col:
                    count = int((df[col].str.strip() == val).sum())
                    return {"value": count, "method": "pandas"}

        elif qtype == "filtered_sum":
            m = re.search(
                r"total\s+([\w\s\(\)\$]+?)\s+for\s+(?:rows\s+where\s+)?([\w\s]+?)\s+(?:is\s+)?['\"](.+?)['\"]",
                question, re.IGNORECASE,
            )
            if m:
                num_hint, cat_hint, val = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                num_col = _find_column(df, num_hint)
                cat_col = _find_column(df, cat_hint)
                if num_col and cat_col:
                    mask = df[cat_col].str.strip() == val
                    s = _to_numeric(df.loc[mask, num_col])
                    return {"value": round(s.sum(), 2), "method": "pandas"}

        elif qtype == "max_row":
            for col in df.columns:
                if col.lower() in q:
                    s = _to_numeric(df[col])
                    if s.notna().any():
                        idx = s.idxmax()
                        row = df.iloc[idx]
                        return {"value": row.to_dict(), "method": "pandas",
                                "entity": row.iloc[0], "num_val": s.iloc[idx]}

        elif qtype == "min_row":
            for col in df.columns:
                if col.lower() in q:
                    s = _to_numeric(df[col])
                    if s.notna().any():
                        idx = s.idxmin()
                        row = df.iloc[idx]
                        return {"value": row.to_dict(), "method": "pandas",
                                "entity": row.iloc[0], "num_val": s.iloc[idx]}

        elif qtype == "rank_top3":
            m = re.search(r"top\s+3\s+(\w[\w\s]*?)\s+entries?\s+by\s+(\w[\w\s\(\)\$]*)", question, re.IGNORECASE)
            if m:
                ent_hint, num_hint = m.group(1).strip(), m.group(2).strip()
                ent_col = _find_column(df, ent_hint)
                num_col = _find_column(df, num_hint)
                if ent_col and num_col:
                    df2 = df.copy()
                    df2["_n"] = _to_numeric(df[num_col])
                    top3 = df2.nlargest(3, "_n")
                    pairs = list(zip(top3[ent_col].tolist(), top3["_n"].tolist()))
                    return {"value": pairs, "method": "pandas",
                            "entities": [e for e, _ in pairs],
                            "num_values": [v for _, v in pairs]}

        elif qtype == "summarization":
            agg = {}
            for col in df.columns:
                s = _to_numeric(df[col])
                if s.notna().sum() >= 2:
                    agg[col] = {"sum": round(s.sum(), 2), "mean": round(s.mean(), 2)}
            first_col_vals = df.iloc[:, 0].tolist() if len(df.columns) > 0 else []
            return {"value": n_rows, "method": "pandas",
                    "row_count": n_rows, "aggregates": agg,
                    "entity_names": first_col_vals}

    except Exception:
        pass

    return {"value": None, "method": "failed"}


# ── Correctness reward (binary) ──────────────────────────────────────────────

def correctness_reward(completion, answer, **kwargs) -> float:
    """Binary correctness: 1.0 if correct, 0.0 if not."""
    qtype = answer.get("question_type", "")
    question = answer.get("instruction", "")
    table_md = answer.get("input", "")
    gold_response = answer.get("response", "")

    text = _get_text(completion)
    pred = _strip_think(text)
    if not pred.strip():
        return 0.0

    gold = _compute_gold(question, table_md, qtype)

    # Math types: exact number match
    if qtype in MATH_TYPES and gold["method"] == "pandas":
        pred_num = _extract_final_number(pred, qtype)
        if pred_num is None:
            return 0.0
        return 1.0 if _approx_equal(pred_num, float(gold["value"])) else 0.0

    # max/min row: must name the correct entity
    if qtype in ("max_row", "min_row") and gold["method"] == "pandas":
        entity = str(gold.get("entity", ""))
        return 1.0 if entity.lower() in pred.lower() else 0.0

    # rank_top3: all 3 entities in correct order
    if qtype == "rank_top3" and gold["method"] == "pandas":
        entities = gold.get("entities", [])
        if not entities:
            return 0.0
        positions = []
        for e in entities:
            pos = pred.lower().find(e.strip().lower())
            if pos < 0:
                return 0.0
            positions.append(pos)
        return 1.0 if positions == sorted(positions) else 0.0

    # summarization: correct row count mentioned
    if qtype == "summarization" and gold.get("method") == "pandas":
        gold_count = gold["row_count"]
        pred_nums = _extract_numbers(pred)
        return 1.0 if any(_approx_equal(pn, gold_count) for pn in pred_nums) else 0.0

    # Fallback: token F1 > 0.8 threshold
    gold_answer = _strip_think(gold_response)
    if gold_answer:
        return 1.0 if _token_f1(pred, gold_answer) >= 0.8 else 0.0
    return 0.0


# ── Environment ──────────────────────────────────────────────────────────────

def load_environment(**kwargs) -> vf.Environment:
    """Entry point for Surogate GRPO."""
    hf_repo = kwargs.pop("hf_repo", "cetusian/markdown-table-qa")
    split = kwargs.pop("split", "train")
    focus_types = kwargs.pop("focus_types", None)
    max_examples = kwargs.pop("max_examples", -1)

    ds = load_dataset(hf_repo, split=split)

    if focus_types:
        if isinstance(focus_types, str):
            focus_types = [t.strip() for t in focus_types.split(",")]
        ds = ds.filter(lambda x: x["question_type"] in set(focus_types))

    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))

    questions = []
    answers = []
    for row in ds:
        questions.append(f"{row['input']}\n\n{row['instruction']}")
        answers.append({
            "instruction": row["instruction"],
            "input": row["input"],
            "response": row["response"],
            "question_type": row["question_type"],
            "domain": row["domain"],
            "id": row["id"],
        })

    env_dataset = HFDataset.from_dict({
        "question": questions,
        "answer": answers,
        "task": ["markdown-table-qa"] * len(questions),
        "info": [{}] * len(questions),
    })

    rubric = vf.Rubric(
        funcs=[correctness_reward],
        weights=[1.0],
    )

    return vf.SingleTurnEnv(
        dataset=env_dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        **kwargs,
    )
