"""Hybrid retrieval: ten dense neighbors, ten BM25 hits, then RRF and a cross-encoder.

Dense search and BM25 each contribute at most ``INITIAL_K`` chunks. Reciprocal
rank fusion uses the usual constant of 60, with ranks starting at 1:

    rrf_score(chunk) = sum(1 / (60 + rank))

A chunk that is missing from a list adds nothing for that list. The fused
chunks are then scored by a cross-encoder, and the top ``k`` of that ranking
are returned. ``score`` is the cross-encoder score.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass

from chromadb.api import ClientAPI
from chromadb.errors import NotFoundError
from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.embed import embed_texts
from src.vectordb import COLLECTION, SearchHit, bm25_scores, get_chunks, search

INITIAL_K = 10
RRF_K = 60
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


def hybrid_search(client: ClientAPI, query: str, k: int = 5) -> list[HybridHit]:
    """Retrieve, fuse, and rerank chunks for ``query``.

    The embedding is the same Ollama call as the index. An empty store or a
    blank query returns nothing and does not embed or rerank.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
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
    return [
        _to_hybrid(by_id[ranked_ids[index]], rerank_scores[index], fused[ranked_ids[index]])
        for index in order[:k]
    ]


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
