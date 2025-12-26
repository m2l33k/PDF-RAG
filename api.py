from __future__ import annotations

import json
import os
import random
import time
import tomllib
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent
STORE_DIR = PROJECT_DIR / "vector_store"

GITHUB_ENDPOINT = "https://models.github.ai/inference"
GITHUB_MODELS_URL = f"{GITHUB_ENDPOINT}/chat/completions"
GITHUB_MODEL_DEFAULT = "meta/Llama-3.3-70B-Instruct"


def get_github_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN")
    if token and token.strip():
        return token.strip()
    secrets_path = Path.home() / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None
    try:
        data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    raw = data.get("GITHUB_TOKEN")
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    language: str = Field(default="English")
    top_k: int = Field(default=6, ge=1, le=50)
    use_reranker: bool = Field(default=False)
    candidate_k: int = Field(default=10, ge=1, le=200)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    filter_rerank: bool = Field(default=False)
    min_rerank_score: float = Field(default=-1.0)
    github_model: str = Field(default=GITHUB_MODEL_DEFAULT)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    timeout_s: int = Field(default=300, ge=10, le=1800)


class SourceHit(BaseModel):
    source: str
    page: int
    score: float
    rank: int
    base_score: float | None = None
    rerank_score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceHit]


class QuizRequest(BaseModel):
    language: str = Field(default="English")
    n_questions: int = Field(default=8, ge=1, le=30)
    difficulty: str = Field(default="medium")
    topic: str | None = Field(default=None)
    source_filter: list[str] = Field(default_factory=list)
    seed: int = Field(default=7)
    top_k: int = Field(default=8, ge=1, le=50)
    use_reranker: bool = Field(default=False)
    candidate_k: int = Field(default=10, ge=1, le=200)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    github_model: str = Field(default=GITHUB_MODEL_DEFAULT)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=768, ge=1, le=4096)
    timeout_s: int = Field(default=300, ge=10, le=1800)


class QuizQuestion(BaseModel):
    question: str
    choices: list[str]
    answer_index: int
    explanation: str | None = None


class QuizSource(BaseModel):
    source: str
    page: int


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]
    sources: list[QuizSource]


def load_store(store_dir: Path) -> tuple[faiss.Index, list[dict[str, Any]], dict[str, Any]]:
    idx = faiss.read_index(str(store_dir / "chunks.faiss"))
    meta = json.loads((store_dir / "chunks.json").read_text(encoding="utf-8"))
    cfg = json.loads((store_dir / "store_config.json").read_text(encoding="utf-8"))
    return idx, meta, cfg


def get_embed_model(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def get_cross_encoder(model_name: str) -> CrossEncoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(model_name, device=device)


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
    scores, ids = idx.search(q_vec, int(top_k))
    out: list[dict[str, Any]] = []
    for rank, (score, i) in enumerate(zip(scores[0].tolist(), ids[0].tolist())):
        if i < 0:
            continue
        row = dict(meta[i])
        row["score"] = float(score)
        row["rank"] = int(rank)
        out.append(row)
    return out


def rerank_cross_encoder(question: str, hits: list[dict[str, Any]], model: CrossEncoder) -> list[dict[str, Any]]:
    if not hits:
        return []
    pairs: list[list[str]] = []
    for h in hits:
        source = str(h.get("source", "")).strip()
        chunk = str(h.get("chunk", "")).strip()
        doc = f"{source}\n{chunk}".strip() if source else chunk
        pairs.append([str(question), doc])
    scores = model.predict(pairs)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    order = np.argsort(-scores)
    out: list[dict[str, Any]] = []
    for rank, j in enumerate(order.tolist()):
        row = dict(hits[int(j)])
        row["base_score"] = float(row.get("score", 0.0))
        row["rerank_score"] = float(scores[int(j)])
        row["score"] = float(scores[int(j)])
        row["rank"] = int(rank)
        out.append(row)
    return out


def dedupe_hits_by_source(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for h in hits:
        src = str(h.get("source", "")).strip()
        if not src:
            out.append(h)
            continue
        if src in seen:
            continue
        seen.add(src)
        out.append(h)
    return out


def build_messages(question: str, contexts: list[dict[str, Any]], language: str) -> list[dict[str, str]]:
    max_context_chars = 4500
    parts: list[str] = []
    used = 0
    for c in contexts:
        part = f"Source: {c['source']} (page {c['page']})\n{c['chunk']}"
        if used + len(part) > max_context_chars:
            break
        parts.append(part)
        used += len(part)
    joined = "\n\n".join(parts)
    system = (
        "You are a helpful tutor. Use only the provided context. "
        "If the context does not contain the answer, say you do not know. "
        f"Answer in {language}."
    )
    user = f"Context:\n{joined}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_quiz_prompt(language: str, n_questions: int, difficulty: str, contexts: list[dict[str, Any]]) -> list[dict[str, str]]:
    max_context_chars = 6000
    parts: list[str] = []
    used = 0
    for c in contexts:
        part = f"Source: {c['source']} (page {c['page']})\n{c['chunk']}"
        if used + len(part) > max_context_chars:
            break
        parts.append(part)
        used += len(part)
    joined = "\n\n".join(parts)
    system = (
        "You create quizzes from study material. "
        f"Write everything in {language}. "
        "Return valid JSON only. Do not use markdown or code fences. Ensure all JSON strings are properly escaped."
    )
    user = (
        f"Material:\n{joined}\n\n"
        f"Create a quiz with {n_questions} multiple-choice questions "
        f"({difficulty} difficulty). Each question must have exactly 4 choices.\n\n"
        "Return JSON with this shape:\n"
        '{ "questions": [ { "question": "...", "choices": ["A","B","C","D"], "answer_index": 0, "explanation": "..." } ] }'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def safe_load_json(text: str) -> dict[str, Any]:
    def strip_fences(s: str) -> str:
        s2 = s.strip()
        if not s2.startswith("```"):
            return s2
        s2 = s2.strip("`")
        parts = s2.split("\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
        return s2.strip()

    def extract_json_substring(s: str) -> str | None:
        s = s.strip()
        start_candidates = [i for i in (s.find("{"), s.find("[")) if i != -1]
        if not start_candidates:
            return None
        start = min(start_candidates)
        in_str = False
        esc = False
        depth = 0
        opening = s[start]
        closing = "}" if opening == "{" else "]"
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return None

    s = strip_fences(text)
    s = (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .strip()
    )
    try:
        return json.loads(s)
    except Exception:
        sub = extract_json_substring(s)
        if sub:
            return json.loads(sub)
        raise


def generate_text_github(
    *,
    messages: list[dict[str, str]],
    model: str,
    token: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> str:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    last_exc: Exception | None = None
    for attempt in range(8):
        try:
            r = requests.post(
                GITHUB_MODELS_URL,
                headers=headers,
                json=payload,
                timeout=int(timeout_s),
            )
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                base_sleep = float(retry_after) if retry_after and retry_after.isdigit() else (1.5 * (2**attempt))
                time.sleep(min(60.0, base_sleep + random.random()))
                continue
            r.raise_for_status()
            data = r.json()
            return str(data["choices"][0]["message"]["content"]).strip()
        except requests.RequestException as e:
            last_exc = e
            time.sleep(min(60.0, (1.0 * (2**attempt)) + random.random()))
    raise RuntimeError("GitHub Models request failed after retries") from last_exc


idx: faiss.Index | None = None
meta: list[dict[str, Any]] | None = None
cfg: dict[str, Any] | None = None
embed_model: SentenceTransformer | None = None

app = FastAPI(title="PDF RAG API", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    global idx, meta, cfg, embed_model
    if not (STORE_DIR / "chunks.faiss").exists():
        raise RuntimeError("Missing vector_store. Build it with pdf_rag_pipeline.ipynb first.")
    idx, meta, cfg = load_store(STORE_DIR)
    embed_model = get_embed_model(str(cfg["embedding_model"]))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "has_token": bool(get_github_token())}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    token = get_github_token()
    if not token:
        raise HTTPException(status_code=500, detail="Missing GITHUB_TOKEN environment variable")
    if idx is None or meta is None or cfg is None or embed_model is None:
        raise HTTPException(status_code=500, detail="Store not loaded")

    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    base_k = int(req.candidate_k) if bool(req.use_reranker) else int(req.top_k)
    base_k = max(int(req.top_k), base_k)

    contexts = retrieve(q, idx, meta, embed_model, top_k=int(base_k))
    if bool(req.use_reranker):
        ce = get_cross_encoder(req.reranker_model.strip())
        contexts = rerank_cross_encoder(q, contexts, ce)
        if bool(req.filter_rerank):
            kept = [c for c in contexts if float(c.get("score", float("-inf"))) >= float(req.min_rerank_score)]
            contexts = kept if kept else contexts
        contexts = dedupe_hits_by_source(contexts)
        for r, c in enumerate(contexts):
            c["rank"] = int(r)
    contexts = contexts[: int(req.top_k)]

    messages = build_messages(q, contexts, req.language)
    answer = generate_text_github(
        messages=messages,
        model=req.github_model.strip(),
        token=token,
        temperature=float(req.temperature),
        max_tokens=int(req.max_tokens),
        timeout_s=int(req.timeout_s),
    )

    sources: list[SourceHit] = []
    for c in contexts:
        sources.append(
            SourceHit(
                source=str(c.get("source", "")),
                page=int(c.get("page", -1)),
                score=float(c.get("score", 0.0)),
                rank=int(c.get("rank", -1)),
                base_score=float(c["base_score"]) if "base_score" in c else None,
                rerank_score=float(c["rerank_score"]) if "rerank_score" in c else None,
            )
        )
    return AskResponse(answer=answer, sources=sources)


@app.post("/quiz", response_model=QuizResponse)
def quiz(req: QuizRequest) -> QuizResponse:
    token = get_github_token()
    if not token:
        raise HTTPException(status_code=500, detail="Missing GITHUB_TOKEN environment variable")
    if idx is None or meta is None or cfg is None or embed_model is None:
        raise HTTPException(status_code=500, detail="Store not loaded")

    topic = (req.topic or "").strip()
    source_filter = [s.strip() for s in (req.source_filter or []) if str(s).strip()]
    source_filter_set = set(source_filter)

    if topic:
        quiz_query = f"Create a quiz about: {topic}"
        base_k = max(int(req.top_k), 8)
        if bool(req.use_reranker):
            base_k = max(base_k, int(req.candidate_k))
        contexts = retrieve(quiz_query, idx, meta, embed_model, top_k=int(base_k))
        if bool(req.use_reranker):
            ce = get_cross_encoder(req.reranker_model.strip())
            contexts = rerank_cross_encoder(quiz_query, contexts, ce)
        contexts = contexts[: max(8, int(req.top_k))]
    else:
        rng = np.random.default_rng(int(req.seed))
        pool = [m for m in meta if m.get("chunk") and m.get("source")]
        if source_filter_set:
            pool = [m for m in pool if str(m.get("source", "")) in source_filter_set]
        take = min(len(pool), max(8, int(req.top_k)))
        picks = rng.choice(len(pool), size=take, replace=False) if take and len(pool) else []
        contexts = [dict(pool[int(i)]) for i in picks]
        for r, c in enumerate(contexts):
            c["rank"] = int(r)
            c["score"] = float("nan")

    if source_filter_set:
        contexts = [c for c in contexts if str(c.get("source", "")) in source_filter_set]

    messages = build_quiz_prompt(req.language, int(req.n_questions), req.difficulty, contexts)
    raw = generate_text_github(
        messages=messages,
        model=req.github_model.strip(),
        token=token,
        temperature=float(req.temperature),
        max_tokens=int(req.max_tokens),
        timeout_s=int(req.timeout_s),
    )
    try:
        quiz_obj = safe_load_json(raw)
    except Exception:
        raise HTTPException(status_code=502, detail="Model returned invalid JSON")

    qs = quiz_obj.get("questions") if isinstance(quiz_obj, dict) else None
    if not isinstance(qs, list):
        raise HTTPException(status_code=502, detail="Model JSON missing 'questions' list")

    questions: list[QuizQuestion] = []
    for q in qs[: int(req.n_questions)]:
        if not isinstance(q, dict):
            continue
        question_text = str(q.get("question", "")).strip()
        choices_val = q.get("choices")
        choices = [str(x) for x in choices_val] if isinstance(choices_val, list) else []
        answer_index = q.get("answer_index")
        try:
            answer_index_int = int(answer_index)
        except Exception:
            answer_index_int = -1
        explanation = q.get("explanation")
        if not question_text or len(choices) != 4 or not (0 <= answer_index_int <= 3):
            continue
        questions.append(
            QuizQuestion(
                question=question_text,
                choices=choices,
                answer_index=answer_index_int,
                explanation=str(explanation).strip() if explanation is not None else None,
            )
        )

    if not questions:
        raise HTTPException(status_code=502, detail="Model returned no valid questions")

    sources: list[QuizSource] = []
    for c in contexts:
        src = str(c.get("source", "")).strip()
        if not src:
            continue
        sources.append(QuizSource(source=src, page=int(c.get("page", -1))))

    return QuizResponse(questions=questions, sources=sources)
