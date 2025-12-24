from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class EvalConfig:
    store_dir: Path
    eval_csv: Path
    out_dir: Path
    top_k: int
    threshold: float | None


def load_store(store_dir: Path) -> tuple[faiss.Index, list[dict[str, Any]], dict[str, Any]]:
    idx = faiss.read_index(str(store_dir / "chunks.faiss"))
    meta = json.loads((store_dir / "chunks.json").read_text(encoding="utf-8"))
    cfg = json.loads((store_dir / "store_config.json").read_text(encoding="utf-8"))
    return idx, meta, cfg


def get_embed_model(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def retrieve(
    question: str,
    idx: faiss.Index,
    meta: list[dict[str, Any]],
    embed_model: SentenceTransformer,
    top_k: int,
) -> list[dict[str, Any]]:
    q_vec = embed_model.encode([question])
    q_vec = np.asarray(q_vec, dtype=np.float32)
    norms = np.linalg.norm(q_vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q_vec = q_vec / norms
    scores, ids = idx.search(q_vec, top_k)
    out: list[dict[str, Any]] = []
    for rank, (score, i) in enumerate(zip(scores[0].tolist(), ids[0].tolist())):
        if i < 0:
            continue
        row = dict(meta[i])
        row["score"] = float(score)
        row["rank"] = rank
        out.append(row)
    return out


def parse_sources(value: Any) -> set[str]:
    if value is None:
        return set()
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return set()
    return {p.strip() for p in s.split(";") if p.strip()}


def precision_at_k(relevance: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rel_k = relevance[:k]
    return float(sum(rel_k)) / float(k)


def recall_at_k(relevance: list[int]) -> float:
    return 1.0 if any(relevance) else 0.0


def mrr_at_k(relevance: list[int]) -> float:
    for i, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / float(i)
    return 0.0


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
    return [p.strip() for p in parts if p.strip()]


def normalize_phrase(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = s.strip(" ,;:.-")
    return s


def looks_too_generic(phrase: str) -> bool:
    p = phrase.strip().lower()
    bad = {
        "application",
        "applications",
        "business",
        "product",
        "document",
        "contain",
        "contains",
        "about",
        "best",
        "return",
        "capabilities",
        "jobs",
        "feature",
        "page",
        "chapter",
    }
    if p in bad:
        return True
    if len(p) < 4:
        return True
    if p.isdigit():
        return True
    return False


def extract_keyphrases(text: str, max_phrases: int) -> list[str]:
    sentences = split_sentences(text)
    phrases: list[str] = []

    patterns = [
        r"\b([A-Z][A-Za-z0-9\-/ ]{2,60})\s+is\s+(?:an|a|the)\b",
        r"\b([A-Z][A-Za-z0-9\-/ ]{2,60})\s+refers\s+to\b",
        r"\b([A-Z][A-Za-z0-9\-/ ]{2,60})\s+means\b",
        r"\b([A-Z][A-Za-z0-9\-/ ]{2,60})\s+consists\s+of\b",
        r"\b([A-Z][A-Za-z0-9\-/ ]{2,60})\s+includes\b",
    ]

    for sent in sentences[:6]:
        for pat in patterns:
            m = re.search(pat, sent)
            if not m:
                continue
            phrase = normalize_phrase(m.group(1))
            if looks_too_generic(phrase):
                continue
            if phrase not in phrases:
                phrases.append(phrase)
                if len(phrases) >= max_phrases:
                    return phrases

    if len(phrases) < max_phrases:
        kws = extract_keywords(text, max_keywords=max_phrases * 2)
        for kw in kws:
            if looks_too_generic(kw):
                continue
            phrase = kw
            if phrase not in phrases:
                phrases.append(phrase)
                if len(phrases) >= max_phrases:
                    break

    return phrases[:max_phrases]


def extract_keywords(text: str, max_keywords: int) -> list[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "you",
        "your",
        "are",
        "can",
        "will",
        "not",
        "have",
        "has",
        "had",
        "into",
        "also",
        "than",
        "then",
        "they",
        "them",
        "their",
        "there",
        "what",
        "when",
        "where",
        "which",
        "who",
        "how",
        "why",
        "use",
        "using",
        "used",
        "one",
        "two",
        "three",
        "more",
        "most",
        "many",
        "such",
        "some",
        "these",
        "those",
        "page",
        "chapter",
        "table",
        "contents",
        "introduction",
        "example",
        "examples",
        "section",
        "figure",
        "data",
        "information",
        "system",
        "systems",
        "computer",
        "science",
        "all",
        "any",
        "each",
        "per",
        "may",
        "might",
        "should",
        "would",
        "could",
        "about",
        "over",
        "under",
        "between",
        "within",
        "without",
        "because",
        "therefore",
        "however",
        "while",
        "during",
        "before",
        "after",
        "same",
        "different",
        "first",
        "second",
        "third",
        "new",
        "old",
        "make",
        "made",
        "makes",
        "get",
        "gets",
        "got",
        "give",
        "given",
        "take",
        "takes",
        "taken",
        "set",
        "sets",
        "using",
        "useful",
        "need",
        "needs",
        "like",
        "also",
        "well",
        "very",
        "much",
        "many",
        "most",
        "into",
        "onto",
    }
    words = [w.lower() for w in tokenize_words(text)]
    counts: dict[str, int] = {}
    for w in words:
        if w in stop:
            continue
        if len(w) < 4:
            continue
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[str] = []
    for w, _ in ranked:
        out.append(w)
        if len(out) >= max_keywords:
            break
    return out


def generate_questions_from_store(
    meta: list[dict[str, Any]],
    out_csv: Path,
    n_questions: int,
    source_regex: str,
    seed: int,
    min_chunk_chars: int,
    resume: bool,
) -> None:
    ensure_parent_dir(out_csv)
    rx = re.compile(source_regex, flags=re.IGNORECASE)
    rng = random.Random(int(seed))
    candidates = [m for m in meta if isinstance(m.get("chunk"), str) and len(str(m.get("chunk"))) >= min_chunk_chars]
    candidates = [m for m in candidates if rx.search(str(m.get("source", "")) or "")]
    if not candidates:
        raise ValueError("No chunks matched source filter. Adjust --source-regex.")
    rng.shuffle(candidates)

    templates = [
        "What is {kw}?",
        "Define {kw}.",
        "Explain {kw}.",
        "How does {kw} work?",
        "Why is {kw} important?",
        "What problem does {kw} solve?",
        "When would you use {kw}?",
        "What is the relationship between {kw} and the rest of the topic?",
    ]

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    if bool(resume) and out_csv.exists():
        try:
            existing = pd.read_csv(out_csv)
            if "question" in existing.columns and "relevant_sources" in existing.columns:
                for _, r in existing.iterrows():
                    q = str(r["question"]).strip().replace("\n", " ")
                    src = str(r["relevant_sources"]).strip()
                    if q and src:
                        rows.append({"question": q, "relevant_sources": src})
                        seen.add(q.lower())
        except Exception:
            rows = []
            seen = set()
    t_i = 0
    for m in candidates:
        if len(rows) >= int(n_questions):
            break
        chunk = str(m.get("chunk", "")).replace("\n", " ").strip()
        src = str(m.get("source", "")).strip()
        if not chunk or not src:
            continue
        kws = extract_keyphrases(chunk, max_phrases=5)
        if not kws:
            continue
        max_per_chunk = 3
        chosen = kws[:]
        rng.shuffle(chosen)
        for kw in chosen[:max_per_chunk]:
            if len(rows) >= int(n_questions):
                break
            if looks_too_generic(kw):
                continue
            q = templates[t_i % len(templates)].format(kw=kw)
            t_i += 1
            q_norm = q.strip().lower()
            if q_norm in seen:
                continue
            seen.add(q_norm)
            rows.append({"question": q.strip(), "relevant_sources": src})

    if len(rows) < int(n_questions):
        more = candidates[:]
        rng.shuffle(more)
        for m in more:
            if len(rows) >= int(n_questions):
                break
            chunk = str(m.get("chunk", "")).replace("\n", " ").strip()
            src = str(m.get("source", "")).strip()
            if not chunk or not src:
                continue
            kws = extract_keywords(chunk, max_keywords=10)
            for kw in kws:
                if len(rows) >= int(n_questions):
                    break
                q = f"What does the passage say about {kw}?"
                q_norm = q.strip().lower()
                if q_norm in seen:
                    continue
                seen.add(q_norm)
                rows.append({"question": q.strip(), "relevant_sources": src})

    if len(rows) < int(n_questions):
        raise ValueError(f"Only generated {len(rows)} questions; decrease --min-chunk-chars or relax --source-regex.")

    pd.DataFrame(rows[: int(n_questions)]).to_csv(out_csv, index=False)


def strip_json_wrappers(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        parts = s.split("\n", 1)
        if len(parts) == 2:
            s = parts[1]
    return s.strip()


def generate_questions_with_github(
    meta: list[dict[str, Any]],
    out_csv: Path,
    n_questions: int,
    source_regex: str,
    seed: int,
    min_chunk_chars: int,
    github_endpoint: str,
    model: str,
    per_call: int,
    timeout_s: int,
    resume: bool,
) -> None:
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("Missing GITHUB_TOKEN environment variable")

    ensure_parent_dir(out_csv)
    rx = re.compile(source_regex, flags=re.IGNORECASE)
    rng = random.Random(int(seed))
    candidates = [m for m in meta if isinstance(m.get("chunk"), str) and len(str(m.get("chunk"))) >= min_chunk_chars]
    candidates = [m for m in candidates if rx.search(str(m.get("source", "")) or "")]
    if not candidates:
        raise ValueError("No chunks matched source filter. Adjust --source-regex.")
    rng.shuffle(candidates)

    out_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    if bool(resume) and out_csv.exists():
        try:
            existing = pd.read_csv(out_csv)
            if "question" in existing.columns and "relevant_sources" in existing.columns:
                for _, r in existing.iterrows():
                    q = str(r["question"]).strip().replace("\n", " ")
                    src = str(r["relevant_sources"]).strip()
                    if q and src:
                        out_rows.append({"question": q, "relevant_sources": src})
                        seen.add(q.lower())
        except Exception:
            out_rows = []
            seen = set()

    url = github_endpoint.rstrip("/") + "/chat/completions"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }

    for m in candidates:
        if len(out_rows) >= int(n_questions):
            break
        chunk = str(m.get("chunk", "")).replace("\n", " ").strip()
        src = str(m.get("source", "")).strip()
        if not chunk or not src:
            continue

        system = (
            "You create high-quality computer science study questions. "
            "Use only the provided passage. "
            "Return valid JSON only."
        )
        user = (
            f"Passage:\n{chunk}\n\n"
            f"Generate {int(per_call)} short questions that are directly answerable from the passage. "
            "Avoid generic questions. Avoid one-word topics. "
            "Return JSON with shape: {\"questions\": [\"...\"]}."
        )

        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.4,
            "max_tokens": 900,
        }

        r = None
        last_exc: Exception | None = None
        for attempt in range(10):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=int(timeout_s))
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    base_sleep = float(retry_after) if retry_after and retry_after.isdigit() else (2.0 * (2**attempt))
                    time.sleep(min(90.0, base_sleep + random.random()))
                    continue
                r.raise_for_status()
                break
            except requests.RequestException as e:
                last_exc = e
                time.sleep(min(90.0, (1.0 * (2**attempt)) + random.random()))
        else:
            raise RuntimeError("GitHub Models question generation failed after retries") from last_exc

        if r is None:
            raise RuntimeError("GitHub Models request failed")
        data = r.json()
        content = str(data["choices"][0]["message"]["content"])
        content = strip_json_wrappers(content)
        obj = json.loads(content)
        qs = obj.get("questions") if isinstance(obj, dict) else None
        if not isinstance(qs, list):
            continue
        for q in qs:
            if len(out_rows) >= int(n_questions):
                break
            if not isinstance(q, str):
                continue
            q2 = q.strip().replace("\n", " ")
            if not q2 or len(q2) < 10:
                continue
            norm = q2.lower()
            if norm in seen:
                continue
            seen.add(norm)
            out_rows.append({"question": q2, "relevant_sources": src})
        pd.DataFrame(out_rows[: int(n_questions)]).to_csv(out_csv, index=False)

    if len(out_rows) < int(n_questions):
        raise ValueError(f"Only generated {len(out_rows)} questions; expand the store or relax --source-regex.")

    pd.DataFrame(out_rows[: int(n_questions)]).to_csv(out_csv, index=False)


def safe_name(value: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("._") or "store"


def store_label(store_dir: Path, store_cfg: dict[str, Any]) -> str:
    parts: list[str] = [store_dir.name]
    if "chunk_size" in store_cfg:
        parts.append(f"cs{store_cfg.get('chunk_size')}")
    if "chunk_overlap" in store_cfg:
        parts.append(f"ov{store_cfg.get('chunk_overlap')}")
    if "embedding_model" in store_cfg:
        parts.append(safe_name(str(store_cfg.get("embedding_model"))).replace("sentence-transformers_", "st_"))
    return safe_name("_".join(parts))


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    thresholds = np.unique(y_score)
    if thresholds.size == 0:
        return 0.0
    best_t = float(thresholds[0])
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not relevant", "relevant"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, out_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close(fig)
    if roc_auc == roc_auc:
        metrics["roc_auc"] = float(roc_auc)

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay(precision=prec, recall=rec, average_precision=ap).plot(ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=200)
    plt.close(fig)
    if ap == ap:
        metrics["average_precision"] = float(ap)

    return metrics


def plot_k_curves(per_query: pd.DataFrame, out_dir: Path, top_k: int) -> None:
    ks = list(range(1, top_k + 1))
    p_means: list[float] = []
    r_means: list[float] = []

    for k in ks:
        p_means.append(float(per_query[f"precision@{k}"].mean()))
        r_means.append(float(per_query[f"recall@{k}"].mean()))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, p_means, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Precision@k (mean)")
    ax.set_title("Precision@k Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "precision_at_k.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, r_means, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k (mean)")
    ax.set_title("Recall@k Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "recall_at_k.png", dpi=200)
    plt.close(fig)


def run_eval(cfg: EvalConfig) -> None:
    ensure_out_dir(cfg.out_dir)
    idx, meta, store_cfg = load_store(cfg.store_dir)
    embed_model = get_embed_model(store_cfg["embedding_model"])

    if not cfg.eval_csv.exists():
        raise FileNotFoundError(
            f"Missing eval CSV: {cfg.eval_csv}. Create it with columns: question,relevant_sources "
            f"(example created at {Path('eval') / 'questions.csv'})."
        )
    df = pd.read_csv(cfg.eval_csv)
    if "question" not in df.columns or "relevant_sources" not in df.columns:
        raise ValueError("CSV must include columns: question, relevant_sources")

    all_y_true: list[int] = []
    all_y_score: list[float] = []
    per_query_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []

    for row_id, row in df.iterrows():
        q = str(row["question"]).strip()
        rel_sources = parse_sources(row["relevant_sources"])
        if not q:
            continue

        hits = retrieve(q, idx, meta, embed_model, top_k=cfg.top_k)
        relevance = [1 if h.get("source") in rel_sources else 0 for h in hits]

        for h, rel in zip(hits, relevance):
            all_y_true.append(int(rel))
            all_y_score.append(float(h.get("score", 0.0)))
            detailed_rows.append(
                {
                    "query_id": int(row_id),
                    "question": q,
                    "hit_rank": int(h.get("rank", -1)),
                    "hit_score": float(h.get("score", 0.0)),
                    "hit_source": str(h.get("source", "")),
                    "hit_page": int(h.get("page", -1)),
                    "is_relevant": int(rel),
                }
            )

        per_query: dict[str, Any] = {"query_id": int(row_id), "question": q}
        for k in range(1, cfg.top_k + 1):
            per_query[f"precision@{k}"] = precision_at_k(relevance, k)
            per_query[f"recall@{k}"] = recall_at_k(relevance[:k])
        per_query["mrr"] = mrr_at_k(relevance)
        per_query_rows.append(per_query)

    if not per_query_rows:
        raise ValueError("No valid questions found in CSV.")

    per_query_df = pd.DataFrame(per_query_rows)
    detailed_df = pd.DataFrame(detailed_rows)
    per_query_df.to_csv(cfg.out_dir / "per_query_metrics.csv", index=False)
    detailed_df.to_csv(cfg.out_dir / "detailed_hits.csv", index=False)

    y_true = np.asarray(all_y_true, dtype=int)
    y_score = np.asarray(all_y_score, dtype=float)

    chosen_t = cfg.threshold if cfg.threshold is not None else find_best_threshold(y_true, y_score)
    y_pred = (y_score >= chosen_t).astype(int)

    plot_confusion(y_true, y_pred, cfg.out_dir / "confusion_matrix.png", f"Confusion Matrix (threshold={chosen_t:.4f})")
    curve_metrics = plot_roc_pr(y_true, y_score, cfg.out_dir)
    plot_k_curves(per_query_df, cfg.out_dir, cfg.top_k)

    summary = {
        "store_dir": str(cfg.store_dir),
        "store_config": store_cfg,
        "top_k": cfg.top_k,
        "threshold": float(chosen_t),
        "mean_mrr": float(per_query_df["mrr"].mean()),
        "mean_precision@k": float(per_query_df[f"precision@{cfg.top_k}"].mean()),
        "mean_recall@k": float(per_query_df[f"recall@{cfg.top_k}"].mean()),
        **curve_metrics,
    }
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved evaluation report to:", str(cfg.out_dir))
    print(json.dumps(summary, indent=2))


def run_sweep(store_dirs: list[Path], eval_csv: Path, out_dir: Path, top_k: int, threshold: float | None) -> None:
    ensure_out_dir(out_dir)
    rows: list[dict[str, Any]] = []
    for store_dir in store_dirs:
        idx, meta, store_cfg = load_store(store_dir)
        label = store_label(store_dir, store_cfg)
        one_out = out_dir / label
        cfg = EvalConfig(
            store_dir=store_dir,
            eval_csv=eval_csv,
            out_dir=one_out,
            top_k=top_k,
            threshold=threshold,
        )
        run_eval(cfg)
        summary_path = one_out / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        flat = {
            "label": label,
            "store_dir": str(store_dir),
            "embedding_model": str(store_cfg.get("embedding_model", "")),
            "normalize": bool(store_cfg.get("normalize", False)),
            "chunk_size": store_cfg.get("chunk_size", ""),
            "chunk_overlap": store_cfg.get("chunk_overlap", ""),
            "top_k": summary.get("top_k", top_k),
            "threshold": summary.get("threshold", ""),
            "mean_mrr": summary.get("mean_mrr", ""),
            "mean_precision@k": summary.get("mean_precision@k", ""),
            "mean_recall@k": summary.get("mean_recall@k", ""),
            "roc_auc": summary.get("roc_auc", ""),
            "average_precision": summary.get("average_precision", ""),
        }
        rows.append(flat)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "aggregate.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--store-dir", default="vector_store", type=str)
    p.add_argument("--store-dirs", default=None, type=str)
    p.add_argument("--store-glob", default=None, type=str)
    p.add_argument("--eval-csv", default="eval/questions.csv", type=str)
    p.add_argument("--out-dir", default="reports/eval", type=str)
    p.add_argument("--top-k", default=6, type=int)
    p.add_argument("--threshold", default=None, type=float)
    p.add_argument("--make-questions", action="store_true")
    p.add_argument("--questions-out", default="eval/questions.csv", type=str)
    p.add_argument("--n-questions", default=200, type=int)
    p.add_argument("--source-regex", default=r"(computer|software|sql|program|coding|interview|faang|cracking|foundation|information|systems)", type=str)
    p.add_argument("--min-chunk-chars", default=220, type=int)
    p.add_argument("--seed", default=7, type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--questions-backend", default="github", choices=["github", "heuristic"], type=str)
    p.add_argument("--github-endpoint", default="https://models.github.ai/inference", type=str)
    p.add_argument("--github-model", default="meta/Llama-3.3-70B-Instruct", type=str)
    p.add_argument("--questions-per-call", default=10, type=int)
    p.add_argument("--github-timeout", default=120, type=int)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    eval_csv = Path(args.eval_csv)
    out_dir = Path(args.out_dir)
    store_dir = Path(args.store_dir)
    if bool(args.make_questions):
        _, meta, _ = load_store(store_dir)
        if str(args.questions_backend) == "github":
            generate_questions_with_github(
                meta=meta,
                out_csv=Path(args.questions_out),
                n_questions=int(args.n_questions),
                source_regex=str(args.source_regex),
                seed=int(args.seed),
                min_chunk_chars=int(args.min_chunk_chars),
                github_endpoint=str(args.github_endpoint),
                model=str(args.github_model),
                per_call=int(args.questions_per_call),
                timeout_s=int(args.github_timeout),
                resume=bool(args.resume),
            )
        else:
            generate_questions_from_store(
                meta=meta,
                out_csv=Path(args.questions_out),
                n_questions=int(args.n_questions),
                source_regex=str(args.source_regex),
                seed=int(args.seed),
                min_chunk_chars=int(args.min_chunk_chars),
                resume=bool(args.resume),
            )
        print("Saved questions to:", str(Path(args.questions_out)))
        return
    if args.store_dirs or args.store_glob:
        store_dirs: list[Path] = []
        if args.store_dirs:
            store_dirs.extend([Path(s.strip()) for s in str(args.store_dirs).split(",") if s.strip()])
        if args.store_glob:
            store_dirs.extend(sorted(Path().glob(str(args.store_glob))))
        store_dirs = [p for p in store_dirs if p.exists()]
        if not store_dirs:
            raise ValueError("No valid store directories found for sweep.")
        run_sweep(
            store_dirs=store_dirs,
            eval_csv=eval_csv,
            out_dir=out_dir,
            top_k=int(args.top_k),
            threshold=args.threshold,
        )
    else:
        cfg = EvalConfig(
            store_dir=store_dir,
            eval_csv=eval_csv,
            out_dir=out_dir,
            top_k=int(args.top_k),
            threshold=args.threshold,
        )
        run_eval(cfg)


if __name__ == "__main__":
    main()
