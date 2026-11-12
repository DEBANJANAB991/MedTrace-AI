"""Optional embedding-based retrieval, separate from the keyword baseline."""

from math import isfinite, sqrt
from typing import Protocol, Sequence

from medtrace.database import SQLiteDatabase
from medtrace.retrieval.keyword import _rank_by_score_and_timestamp
from medtrace.retrieval.models import RankedEvidence


class TextEmbedder(Protocol):
    """Produces one numeric vector for each input text."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


class SentenceTransformerEmbedder:
    """Lazy adapter for the optional sentence-transformers package."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Semantic retrieval requires the optional dependency. "
                "Install it with: pip install -e \".[semantic]\""
            ) from error

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = self._model.encode(list(texts), convert_to_numpy=True)
        return vectors.tolist()


def _cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Embedding vectors must have matching dimensions")
    if not first:
        raise ValueError("Embedding vectors cannot be empty")

    first_values = [float(value) for value in first]
    second_values = [float(value) for value in second]
    if not all(isfinite(value) for value in (*first_values, *second_values)):
        raise ValueError("Embedding vectors must contain only finite numbers")

    first_norm = sqrt(sum(value * value for value in first_values))
    second_norm = sqrt(sum(value * value for value in second_values))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    dot_product = sum(
        first_value * second_value
        for first_value, second_value in zip(first_values, second_values)
    )
    similarity = dot_product / (first_norm * second_norm)
    return max(-1.0, min(1.0, similarity))


class SemanticRetriever:
    """Rank patient evidence by cosine similarity of text embeddings."""

    def __init__(
        self,
        database: SQLiteDatabase,
        embedder: TextEmbedder,
        max_records: int = 5,
        min_similarity: float = 0.0,
    ) -> None:
        if max_records < 1:
            raise ValueError("max_records must be at least 1")
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1.0 and 1.0")
        self._database = database
        self._embedder = embedder
        self.max_records = max_records
        self.min_similarity = min_similarity

    def retrieve(self, patient_id: str, query: str) -> list[RankedEvidence]:
        if not query.strip():
            return []

        patient_evidence = self._database.fetch_patient_evidence(patient_id)
        if not patient_evidence:
            return []

        texts = [query, *(record.content for record in patient_evidence)]
        embeddings = self._embedder.encode(texts)
        if len(embeddings) != len(texts):
            raise ValueError(
                "Embedder must return exactly one vector for each input text"
            )

        query_embedding = embeddings[0]
        ranked = [
            RankedEvidence(
                evidence=record,
                score=_cosine_similarity(query_embedding, embedding),
                retrieval_method="semantic",
            )
            for record, embedding in zip(patient_evidence, embeddings[1:])
        ]
        ranked = [
            result for result in ranked if result.score > self.min_similarity
        ]
        return _rank_by_score_and_timestamp(ranked)[: self.max_records]
