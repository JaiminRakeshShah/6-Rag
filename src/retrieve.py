"""Hybrid retrieval: ten dense neighbors, ten BM25 hits, then RRF, a cross-encoder, and Jev.

Dense search and BM25 each contribute at most ``INITIAL_K`` chunks. Reciprocal
rank fusion uses the usual constant of 60, with ranks starting at 1:

    rrf_score(chunk) = sum(1 / (60 + rank))

A chunk that is missing from a list adds nothing for that list. The fused
chunks are scored by a cross-encoder. Every fused chunk is then one Choice
option for Jev. Jev returns a probability for each option. The three highest
probabilities are kept. ``score`` is the cross-encoder score. ``probability``
is the Jev probability.
"""

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from chromadb.api import ClientAPI
from chromadb.errors import NotFoundError
from fastembed.rerank.cross_encoder import TextCrossEncoder
import requests

from src.embed import embed_texts
from src.vectordb import COLLECTION, SearchHit, bm25_scores, get_chunks, search

INITIAL_K = 10
RRF_K = 60
JEV_KEEP = 3
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
_JEV_INSTRUCTIONS = (
    "Which chunk best answers the question? "
    "Use superseded, document_date, and version when the question asks for the current or the archived copy."
)
_DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
_cross_encoder_model: TextCrossEncoder | None = None


@dataclass(frozen=True)
class HybridHit:
    """One reranked hit. ``score`` is the cross-encoder score, higher is better."""

    chunk_id: str
    filename: str
    title: str
    page_no: int | None
    section_name: str
    parent_id: str
    start_span: int | None
    end_span: int | None
    document_date: str | None
    date_source: str | None
    version: str | None
    superseded: bool
    body: str
    score: float
    rrf_score: float
    probability: float = 0.0


def hybrid_search(client: ClientAPI, query: str, k: int = 5) -> list[HybridHit]:
    """Retrieve, fuse, rerank, then keep the three chunks Jev rates highest.

    The embedding is the same Ollama call as the index. An empty store or a
    blank query returns nothing and does not embed, rerank, or call Jev.
    ``k`` still has to be at least 1. The returned list is never longer than
    ``JEV_KEEP``. ``hybrid_search.last_jev`` holds the latest Choice result,
    including a probability for every chunk.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    hybrid_search.last_jev = {}
    count = _indexed_count(client)
    if count == 0 or not query.strip():
        return []

    vector = embed_texts([query])[0]
    dense_hits = search(client, vector, k=min(INITIAL_K, count))
    sparse_ids = _top_ids(bm25_scores(client, query), INITIAL_K)
    fused = reciprocal_rank_fusion([hit.chunk_id for hit in dense_hits], sparse_ids)
    by_id = {hit.chunk_id: hit for hit in dense_hits}
    missing = [chunk_id for chunk_id in fused if chunk_id not in by_id]
    for hit in get_chunks(client, missing):
        by_id[hit.chunk_id] = hit

    ranked_ids = sorted(
        (chunk_id for chunk_id in fused if chunk_id in by_id),
        key=lambda chunk_id: (-fused[chunk_id], chunk_id),
    )
    passages = [by_id[chunk_id].body for chunk_id in ranked_ids]
    rerank_scores = cross_encoder_scores(query, passages)
    order = sorted(
        range(len(ranked_ids)),
        key=lambda index: (
            -rerank_scores[index],
            -fused[ranked_ids[index]],
            ranked_ids[index],
        ),
    )
    candidates = [
        _to_hybrid(by_id[ranked_ids[index]], rerank_scores[index], fused[ranked_ids[index]])
        for index in order
    ]
    if not candidates:
        return []
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TYPESAFE_API_KEY is required to rank chunks with Jev")
    result = call_jev(api_key, query, candidates)
    hybrid_search.last_jev = result
    return _top_probabilities(candidates, result["probabilities"], limit=min(k, JEV_KEEP))


def jev_payload(question: str, hits: Sequence[HybridHit]) -> dict[str, object]:
    """One Choice question whose options are the retrieved chunks."""
    return {
        "model": "jev-latest",
        "state": {"question": question},
        "questions": {
            "action": {
                "type": "choice",
                "instructions": _JEV_INSTRUCTIONS,
                "criteria": {hit.chunk_id: _chunk_criterion(hit) for hit in hits},
            }
        },
    }


def call_jev(api_key: str, question: str, hits: Sequence[HybridHit]) -> dict[str, object]:
    """Ask Jev which chunk answers ``question``. Probabilities cover every chunk."""
    started = time.perf_counter()
    res = requests.post(
        TYPESAFE_URL,
        data=json.dumps(jev_payload(question, hits)).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    res.raise_for_status()
    body = res.json()
    answer = body["answers"]["action"]
    usage = body.get("usage") or {}
    probabilities = answer.get("probabilities") or {}
    if not isinstance(probabilities, dict):
        raise ValueError("Jev choice answer has no probabilities")
    return {
        "router": "jev",
        "output": "structured choice",
        "tool": answer.get("choice"),
        "raw": answer,
        "probabilities": {str(key): float(value) for key, value in probabilities.items()},
        "generation_tokens": usage.get("output_tokens"),
        "prompt_tokens": usage.get("input_tokens"),
        "latency_ms": latency_ms,
    }


def _top_probabilities(
    hits: Sequence[HybridHit],
    probabilities: Mapping[str, float],
    limit: int,
) -> list[HybridHit]:
    """Highest Jev probability first. Ties keep the stronger cross-encoder score."""
    ranked = sorted(
        hits,
        key=lambda hit: (-probabilities.get(hit.chunk_id, 0.0), -hit.score, hit.chunk_id),
    )
    return [
        replace(hit, probability=float(probabilities.get(hit.chunk_id, 0.0)))
        for hit in ranked[:limit]
    ]


def chunk_metadata(hit: HybridHit) -> dict[str, object]:
    """Chunk fields the ranker needs. ``superseded`` is always set. Empty metadata is left out."""
    fields: dict[str, object] = {
        "section_id": hit.parent_id,
        "section_name": hit.section_name,
        "filename": hit.filename,
        "superseded": hit.superseded,
    }
    if hit.title:
        fields["title"] = hit.title
    if hit.page_no is not None:
        fields["page_no"] = hit.page_no
    if hit.document_date is not None:
        fields["document_date"] = hit.document_date
    if hit.date_source is not None:
        fields["date_source"] = hit.date_source
    if hit.version is not None:
        fields["version"] = hit.version
    return fields


def _chunk_criterion(hit: HybridHit) -> dict[str, object]:
    """Description Jev sees for one chunk option."""
    return {**chunk_metadata(hit), "body": hit.body}


def reciprocal_rank_fusion(*rankings: Sequence[str]) -> dict[str, float]:
    """Sum ``1 / (RRF_K + rank)`` across lists. Ranks start at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def cross_encoder_scores(query: str, passages: list[str]) -> list[float]:
    """Score each passage against ``query``. The list matches ``passages`` in order.

    The model is ``RERANK_MODEL``, or ``Xenova/ms-marco-MiniLM-L-6-v2``.
    An empty passage list does not load the model.
    """
    if not passages:
        return []
    model = _cross_encoder()
    return [float(score) for score in model.rerank(query, passages)]


def _cross_encoder() -> TextCrossEncoder:
    """Load the reranker once and reuse it."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        model_id = os.getenv("RERANK_MODEL", _DEFAULT_RERANK_MODEL)
        _cross_encoder_model = TextCrossEncoder(model_name=model_id)
    return _cross_encoder_model


def _indexed_count(client: ClientAPI) -> int:
    """How many chunks the Chroma collection holds. Missing means zero."""
    try:
        collection = client.get_collection(COLLECTION, embedding_function=None)
    except NotFoundError:
        return 0
    return collection.count()


def _top_ids(scores: dict[str, float], limit: int) -> list[str]:
    """Highest scores first, then ``chunk_id``, cut to ``limit``."""
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return ranked[:limit]


def _to_hybrid(hit: SearchHit, score: float, rrf_score: float) -> HybridHit:
    """Copy one stored chunk onto a reranked hit."""
    return HybridHit(
        chunk_id=hit.chunk_id,
        filename=hit.filename,
        title=hit.title,
        page_no=hit.page_no,
        section_name=hit.section_name,
        parent_id=hit.parent_id,
        start_span=hit.start_span,
        end_span=hit.end_span,
        document_date=hit.document_date,
        date_source=hit.date_source,
        version=hit.version,
        superseded=hit.superseded,
        body=hit.body,
        score=score,
        rrf_score=rrf_score,
    )
