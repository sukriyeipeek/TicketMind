"""Geliştirme fixture'larında retrieval ölçümü; model doğruluk iddiası değildir."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from ticketmind.data import PROJECT_ROOT
from ticketmind.rag import ModelUnavailable, RagService


DEFAULT_EVAL = PROJECT_ROOT / "data" / "rag_eval.jsonl"


def evaluate_rag(service: RagService, dataset: Path = DEFAULT_EVAL, top_k: int = 3,
                 with_drafts: bool = False) -> dict:
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or len(rows) > 1000:
        raise ValueError("RAG fixture kümesi 1–1000 sorgu içermeli.")
    known_ids = {doc["id"] for doc in service._documents}
    seen = set()
    results = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or row["id"] in seen or not isinstance(row.get("question"), str)
                or not isinstance(row.get("relevant_ids"), list)
                or any(not isinstance(source_id, str) or source_id not in known_ids
                       for source_id in row["relevant_ids"])
                or len(set(row["relevant_ids"])) != len(row["relevant_ids"])):
            raise ValueError("Geçersiz fixture veya kaynak etiketi.")
        seen.add(row["id"])
        started = perf_counter()
        unavailable = False
        try:
            sources = service.retrieve(row["question"], top_k=top_k)
        except ModelUnavailable:
            sources, unavailable = [], True
        elapsed_ms = (perf_counter() - started) * 1000
        returned = [source["id"] for source in sources]
        gold = set(row["relevant_ids"])
        first_rank = next((rank for rank, source_id in enumerate(returned, 1) if source_id in gold), None)
        result = {
            "id": row["id"], "question": row["question"], "relevant_ids": row["relevant_ids"],
            "returned_ids": returned, "model_unavailable": unavailable,
            "recall_at_k": len(gold & set(returned)) / len(gold) if gold else None,
            "reciprocal_rank": 1 / first_rank if first_rank else 0.0 if gold else None,
            "correct_no_evidence": not returned and not unavailable if not gold else None,
            "retrieval_elapsed_ms": round(elapsed_ms, 3),
        }
        if with_drafts:
            # Durum raporlanır; burada LLM'nin semantik doğruluğu ölçülmez.
            draft = service.draft(row["question"])
            result["draft_status"] = draft["status"]
            result["draft_mode"] = draft["mode"]
        results.append(result)
    answerable = [result for result in results if result["relevant_ids"]]
    unanswerable = [result for result in results if not result["relevant_ids"]]
    return {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "authored_development_fixture",
        "evaluation_role": "development_fixture_not_holdout",
        "model_quality_claim": False,
        "notice": "FAQ ve sorgular birlikte geliştirilmiştir; gerçek kullanıcı veya bağımsız test başarısı değildir.",
        "provider": service.provider,
        "embedding_model": service.embed_model if service.provider == "ollama" else None,
        "generator": service.generator if with_drafts else None,
        "generation_model": service.generation_model if with_drafts and service.generator == "ollama" else None,
        "corpus_sha256": service.corpus_hash,
        "top_k": top_k,
        "n_queries": len(results), "n_answerable": len(answerable), "n_unanswerable": len(unanswerable),
        "recall_at_k": sum(result["recall_at_k"] for result in answerable) / len(answerable) if answerable else None,
        "mrr": sum(result["reciprocal_rank"] for result in answerable) / len(answerable) if answerable else None,
        "no_evidence_accuracy": sum(result["correct_no_evidence"] for result in unanswerable) / len(unanswerable) if unanswerable else None,
        "model_unavailable_count": sum(result["model_unavailable"] for result in results),
        "draft_status_counts": dict(Counter(result["draft_status"] for result in results)) if with_drafts else None,
        "queries": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--knowledge", type=Path, default=None)
    parser.add_argument("--provider", choices=("tfidf", "ollama"), default="tfidf")
    parser.add_argument("--generator", choices=("extractive", "ollama"), default="extractive")
    parser.add_argument("--with-drafts", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "rag-fixture.json")
    args = parser.parse_args()
    service = RagService(provider=args.provider, generator=args.generator,
                         docs_path=args.knowledge, timeout=args.timeout)
    report = evaluate_rag(service, args.dataset, args.top_k, args.with_drafts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "queries"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
