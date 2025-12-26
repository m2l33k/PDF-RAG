from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests
import streamlit as st
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent
STORE_DIR = PROJECT_DIR / "vector_store"

GITHUB_ENDPOINT = "https://models.github.ai/inference"
GITHUB_MODELS_URL = f"{GITHUB_ENDPOINT}/chat/completions"
GITHUB_MODEL_DEFAULT = "meta/Llama-3.3-70B-Instruct"


def load_store(store_dir: Path) -> tuple[faiss.Index, list[dict[str, Any]], dict[str, Any]]:
    idx = faiss.read_index(str(store_dir / "chunks.faiss"))
    meta = json.loads((store_dir / "chunks.json").read_text(encoding="utf-8"))
    cfg = json.loads((store_dir / "store_config.json").read_text(encoding="utf-8"))
    return idx, meta, cfg


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


def get_github_token() -> str | None:
    try:
        token = st.secrets.get("GITHUB_TOKEN")
        if token:
            return str(token)
    except Exception:
        pass
    token = os.environ.get("GITHUB_TOKEN")
    return token.strip() if token else None


def generate_text_github(
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
            break
        except requests.RequestException as e:
            last_exc = e
            time.sleep(min(60.0, (1.0 * (2**attempt)) + random.random()))
    else:
        raise RuntimeError("GitHub Models request failed after retries") from last_exc

    data = r.json()
    return str(data["choices"][0]["message"]["content"]).strip()


@st.cache_resource
def get_embed_model(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


@st.cache_resource
def get_cross_encoder(model_name: str) -> CrossEncoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(model_name, device=device)


@st.cache_resource
def get_store() -> tuple[faiss.Index, list[dict[str, Any]], dict[str, Any]]:
    return load_store(STORE_DIR)


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


st.set_page_config(page_title="PDF RAG", layout="wide")
st.title("PDF RAG (FAISS + Generator)")

if not (STORE_DIR / "chunks.faiss").exists():
    st.error("Missing `vector_store/`. Run `pdf_rag_pipeline.ipynb` to build it first.")
    st.stop()

idx, meta, cfg = get_store()
embed_model = get_embed_model(cfg["embedding_model"])

all_sources = sorted({str(m.get("source", "")) for m in meta if m.get("source")})

with st.sidebar:
    st.subheader("Settings")
    language = st.selectbox("Answer language", ["English", "Français", "العربية"], index=0)
    top_k = st.slider("top_k", min_value=1, max_value=12, value=6, step=1)
    use_reranker = st.checkbox("Use re-ranker (cross-encoder)", value=False)
    candidate_k = st.slider("candidate_k", min_value=top_k, max_value=30, value=max(int(top_k), 10), step=1)
    reranker_model = st.text_input("Re-ranker model", value="cross-encoder/ms-marco-MiniLM-L-6-v2")
    filter_rerank = st.checkbox("Filter re-ranked hits by score", value=False)
    min_rerank_score = st.number_input("Min re-rank score", value=-1.0, step=0.1, disabled=not bool(filter_rerank))
    temperature = st.slider("temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    timeout_s = st.number_input("timeout (seconds)", min_value=30, max_value=600, value=300, step=10)
    github_model = st.text_input("GitHub model", value=GITHUB_MODEL_DEFAULT)
    max_tokens = st.number_input("max_tokens", min_value=32, max_value=2048, value=512, step=32)

tab_ask, tab_quiz = st.tabs(["Ask", "Quiz"])

with tab_ask:
    question = st.text_input("Question", value="", placeholder="Ask something about your PDFs...")
    col1, col2 = st.columns([1, 1])
    with col1:
        ask = st.button("Ask", type="primary")
    with col2:
        st.write(f"GPU: {torch.cuda.is_available()}")

    if ask:
        q = question.strip()
        if not q:
            st.warning("Type a question first.")
            st.stop()
        with st.spinner("Retrieving context..."):
            if bool(use_reranker):
                ce = get_cross_encoder(reranker_model.strip())
                base = retrieve(q, idx, meta, embed_model, top_k=int(candidate_k))
                contexts = rerank_cross_encoder(q, base, ce)
                if bool(filter_rerank):
                    kept = [c for c in contexts if float(c.get("score", float("-inf"))) >= float(min_rerank_score)]
                    contexts = kept if kept else contexts
                contexts = dedupe_hits_by_source(contexts)
                for r, c in enumerate(contexts):
                    c["rank"] = int(r)
                contexts = contexts[: int(top_k)]
            else:
                contexts = retrieve(q, idx, meta, embed_model, top_k=int(top_k))
        with st.spinner("Generating answer..."):
            token = get_github_token()
            if not token:
                st.error("Missing `GITHUB_TOKEN`. Set it as an environment variable or Streamlit secret.")
                st.stop()
            messages = build_messages(q, contexts, language)
            answer = generate_text_github(
                messages=messages,
                model=github_model.strip(),
                token=token,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                timeout_s=int(timeout_s),
            )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for c in contexts:
            if "base_score" in c:
                st.write(
                    f"- {c['source']} (page {c['page']}, sim {float(c['base_score']):.3f}, rerank {float(c['score']):.3f})"
                )
            else:
                st.write(f"- {c['source']} (page {c['page']}, score {c['score']:.3f})")


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


with tab_quiz:
    st.subheader("Quiz Generator")
    quiz_language = st.selectbox("Quiz language", ["English", "Français", "العربية"], index=0, key="quiz_language")
    difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    n_questions = st.slider("Number of questions", min_value=3, max_value=15, value=8, step=1)
    source_filter = st.multiselect("Limit to PDFs (optional)", options=all_sources)
    topic = st.text_input("Topic (optional)", value="", placeholder="e.g. operating systems, databases, networking")
    gen_quiz = st.button("Generate Quiz", type="primary")

    if "quiz" not in st.session_state:
        st.session_state["quiz"] = None
        st.session_state["quiz_sources"] = None

    if gen_quiz:
        token = get_github_token()
        if not token:
            st.error("Missing `GITHUB_TOKEN`. Set it as an environment variable or Streamlit secret.")
            st.stop()

        if topic.strip():
            quiz_query = f"Create a quiz about: {topic.strip()}"
            if bool(use_reranker):
                ce = get_cross_encoder(reranker_model.strip())
                base = retrieve(quiz_query, idx, meta, embed_model, top_k=max(int(candidate_k), 8))
                base_contexts = rerank_cross_encoder(quiz_query, base, ce)[: max(8, int(top_k))]
            else:
                base_contexts = retrieve(quiz_query, idx, meta, embed_model, top_k=max(8, int(top_k)))
        else:
            rng = np.random.default_rng(7)
            pool = [m for m in meta if m.get("chunk") and m.get("source")]
            if source_filter:
                pool = [m for m in pool if m.get("source") in set(source_filter)]
            take = min(len(pool), max(8, int(top_k)))
            picks = rng.choice(len(pool), size=take, replace=False) if take and len(pool) else []
            base_contexts = [dict(pool[int(i)]) for i in picks]
            for r, c in enumerate(base_contexts):
                c["rank"] = r
                c["score"] = float("nan")

        if source_filter:
            base_contexts = [c for c in base_contexts if c.get("source") in set(source_filter)]

        with st.spinner("Generating quiz..."):
            messages = build_quiz_prompt(quiz_language, int(n_questions), difficulty, base_contexts)
            raw = generate_text_github(
                messages=messages,
                model=github_model.strip(),
                token=token,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                timeout_s=int(timeout_s),
            )
            try:
                quiz = safe_load_json(raw)
            except Exception:
                st.error("Quiz generation failed: model returned invalid JSON.")
                st.text("Raw response:")
                st.code(raw)
                st.stop()
            st.session_state["quiz"] = quiz
            st.session_state["quiz_sources"] = base_contexts

    quiz = st.session_state.get("quiz")
    quiz_sources = st.session_state.get("quiz_sources")
    if quiz and isinstance(quiz, dict) and isinstance(quiz.get("questions"), list):
        questions = quiz["questions"]
        st.write(f"Questions: {len(questions)}")
        answers: list[int | None] = []
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}. {q.get('question','')}**")
            choices = q.get("choices") or []
            selected = st.radio(
                label="",
                options=list(range(len(choices))),
                format_func=lambda idx: str(choices[idx]),
                index=None,
                key=f"quiz_q_{i}",
            )
            answers.append(selected)
        submit = st.button("Submit Quiz")
        if submit:
            score = 0
            for i, q in enumerate(questions):
                correct = int(q.get("answer_index", -1))
                if answers[i] is not None and int(answers[i]) == correct:
                    score += 1
            st.success(f"Score: {score}/{len(questions)}")
            for i, q in enumerate(questions):
                st.write(f"Q{i+1} explanation: {q.get('explanation','')}")

        if quiz_sources:
            st.subheader("Material Sources Used")
            for c in quiz_sources:
                st.write(f"- {c['source']} (page {c['page']})")
