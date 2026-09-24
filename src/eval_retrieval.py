"""Retrieval metrics for the fused hybrid ranker on the golden question set.

A chunk is relevant when its section is in ``supporting_sections``. If that
section sets ``chunk_index``, only that chunk counts. Questions marked
``abstain`` have no relevant chunk, so they are left out of the ranking
means. They are still timed.

``k`` is 10, the same width as the initial dense and BM25 lists. The ranked
list is the fused result ``hybrid_search`` returns after reciprocal rank
fusion and the cross-encoder. Precision is the mean of relevant hits divided
by the length of that list. Precision@k divides by ``k`` instead, so a short
list scores lower.
"""

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.chunking import Chunk, chunk_directory
from src.retrieve import INITIAL_K, hybrid_search
from src.vectordb import connect

K = INITIAL_K
ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data" / "gold" / "policy_rag_golden.json"


@dataclass(frozen=True)
class QueryRanking:
    """One scored question: the fused order, the relevant ids, and the wait."""

    retrieved: tuple[str, ...]
    relevant: frozenset[str]
    latency_seconds: float


@dataclass(frozen=True)
class RetrievalEval:
    """Macro averages. Latency is the mean over every timed question."""

    mrr: float
    precision: float
    ndcg: float
    latency_seconds: float
    recall_at_k: float
    precision_at_k: float
    k: int
    scored_questions: int
    timed_questions: int


def relevant_chunk_ids(item: Mapping[str, object], chunks_by_section: Mapping[str, Sequence[Chunk]]) -> set[str]:
    """Chunk ids that count as relevant for one golden question."""
    found: set[str] = set()
    sections = item["supporting_sections"]
    if not isinstance(sections, list):
        raise ValueError("supporting_sections must be a list")
    for section in sections:
        if not isinstance(section, Mapping):
            raise ValueError("supporting section must be an object")
        section_id = section["section_id"]
        if not isinstance(section_id, str):
            raise ValueError("section_id must be a string")
        if "chunk_index" in section:
            found.add(f"{section_id}:{section['chunk_index']}")
            continue
        chunks = chunks_by_section.get(section_id)
        if not chunks:
            raise KeyError(section_id)
        for chunk in chunks:
            found.add(f"{section_id}:{chunk.index}")
    return found


def score_ranking(retrieved: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    """Score one ranked list. Ranks start at 1. Relevance is binary."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        raise ValueError("relevance set is empty")
    top = list(retrieved[:k])
    reciprocal = 0.0
    for rank, chunk_id in enumerate(top, start=1):
        if chunk_id in relevant:
            reciprocal = 1.0 / rank
            break
    hits = sum(1 for chunk_id in top if chunk_id in relevant)
    found = {chunk_id for chunk_id in top if chunk_id in relevant}
    precision = 0.0 if not top else hits / len(top)
    return {
        "reciprocal_rank": reciprocal,
        "precision": precision,
        "ndcg": _ndcg(top, relevant, k),
        "recall_at_k": len(found) / len(relevant),
        "precision_at_k": hits / k,
    }


def evaluate(scored: Sequence[QueryRanking], latencies: Sequence[float], k: int = K) -> RetrievalEval:
    """Average per-question scores. ``latencies`` may include unscored questions."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not scored:
        raise ValueError("no scored questions")
    if not latencies:
        raise ValueError("no timed questions")
    totals = {
        "reciprocal_rank": 0.0,
        "precision": 0.0,
        "ndcg": 0.0,
        "recall_at_k": 0.0,
        "precision_at_k": 0.0,
    }
    for query in scored:
        row = score_ranking(query.retrieved, set(query.relevant), k)
        for key in totals:
            totals[key] += row[key]
    count = len(scored)
    return RetrievalEval(
        mrr=totals["reciprocal_rank"] / count,
        precision=totals["precision"] / count,
        ndcg=totals["ndcg"] / count,
        latency_seconds=sum(latencies) / len(latencies),
        recall_at_k=totals["recall_at_k"] / count,
        precision_at_k=totals["precision_at_k"] / count,
        k=k,
        scored_questions=count,
        timed_questions=len(latencies),
    )


def run_golden(k: int = K) -> RetrievalEval:
    """Search each golden question and average the fused ranking."""
    dataset = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunk_directory():
        grouped.setdefault(chunk.section_id, []).append(chunk)
    client = connect()
    try:
        scored: list[QueryRanking] = []
        latencies: list[float] = []
        for item in dataset["items"]:
            started = time.perf_counter()
            hits = hybrid_search(client, item["question"], k=k)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            if item["abstain"]:
                continue
            scored.append(
                QueryRanking(
                    retrieved=tuple(hit.chunk_id for hit in hits),
                    relevant=frozenset(relevant_chunk_ids(item, grouped)),
                    latency_seconds=elapsed,
                )
            )
    finally:
        client.close()
    return evaluate(scored, latencies, k=k)


def _ndcg(top: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary nDCG. The ideal list puts a relevant chunk in every slot it can fill."""
    dcg = 0.0
    for rank, chunk_id in enumerate(top, start=1):
        if chunk_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def main() -> None:
    """Print the fused retrieval metrics for the golden set."""
    result = run_golden()
    print(f"fused retrieval at k={result.k}")
    print(f"scored questions: {result.scored_questions}")
    print(f"timed questions: {result.timed_questions}")
    print(f"mrr: {result.mrr:.4f}")
    print(f"precision: {result.precision:.4f}")
    print(f"ndcg: {result.ndcg:.4f}")
    print(f"latency_seconds: {result.latency_seconds:.4f}")
    print(f"recall@{result.k}: {result.recall_at_k:.4f}")
    print(f"precision@{result.k}: {result.precision_at_k:.4f}")


if __name__ == "__main__":
    main()
