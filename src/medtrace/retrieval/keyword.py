"""Transparent patient-scoped keyword retrieval baseline."""

import re

from medtrace.database import SQLiteDatabase
from medtrace.retrieval.models import RankedEvidence


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _rank_by_score_and_timestamp(
    results: list[RankedEvidence],
) -> list[RankedEvidence]:
    results.sort(key=lambda result: result.evidence.timestamp, reverse=True)
    results.sort(key=lambda result: result.score, reverse=True)
    return results


class KeywordRetriever:
    """Rank patient evidence by the fraction of query terms it contains."""

    def __init__(self, database: SQLiteDatabase, max_records: int = 5) -> None:
        if max_records < 1:
            raise ValueError("max_records must be at least 1")
        self._database = database
        self.max_records = max_records

    def retrieve(self, patient_id: str, query: str) -> list[RankedEvidence]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        patient_evidence = self._database.fetch_patient_evidence(patient_id)
        ranked: list[RankedEvidence] = []
        for evidence in patient_evidence:
            matched_terms = query_tokens.intersection(_tokens(evidence.content))
            if not matched_terms:
                continue
            ranked.append(
                RankedEvidence(
                    evidence=evidence,
                    score=len(matched_terms) / len(query_tokens),
                    retrieval_method="keyword",
                )
            )

        return _rank_by_score_and_timestamp(ranked)[: self.max_records]
