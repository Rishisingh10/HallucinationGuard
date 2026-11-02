
import json
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from hallucination_guard.claim_aware_reranker import (
    ClaimAwareReranker,
)


class EvidenceRetriever:
    def __init__(
        self,
        corpus_path: str,
        embedding_model: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",
        rrf_k: int = 60,
    ):
        self.corpus_path = Path(corpus_path)
        self.rrf_k = rrf_k

        self.documents = self._load_documents()

        self.tokenized_documents = [
            self._tokenize(document["text"])
            for document in self.documents
        ]

        self.index = (
            BM25Okapi(self.tokenized_documents)
            if self.tokenized_documents
            else None
        )

        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.document_embeddings = None

        self.reranker = ClaimAwareReranker()

    def _load_documents(self):
        with self.corpus_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _tokenize(self, text: str):
        return text.lower().split()

    def _load_embeddings(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device="cpu",
            )

            texts = [
                document["text"]
                for document in self.documents
            ]

            self.document_embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    def _bm25_ranking(self, query: str):
        if self.index is None:
            return []

        scores = self.index.get_scores(
            self._tokenize(query)
        )

        return sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

    def _semantic_ranking(self, query: str):
        self._load_embeddings()

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarities = cosine_similarity(
            query_embedding,
            self.document_embeddings,
        )[0]

        return sorted(
            range(len(similarities)),
            key=lambda index: similarities[index],
            reverse=True,
        )

    def _reciprocal_rank_fusion(self, rankings):
        fused_scores = {}

        for ranking in rankings:
            for rank, document_index in enumerate(
                ranking,
                start=1,
            ):
                score = 1.0 / (self.rrf_k + rank)

                fused_scores[document_index] = (
                    fused_scores.get(document_index, 0.0)
                    + score
                )

        return sorted(
            fused_scores,
            key=lambda index: fused_scores[index],
            reverse=True,
        ), fused_scores

    def retrieve(self, query: str, top_k: int = 3, offset: int = 0):
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if offset < 0:
            raise ValueError("offset cannot be negative.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not self.documents:
            return []

        bm25_ranking = self._bm25_ranking(query)
        semantic_ranking = self._semantic_ranking(query)

        fused_ranking, fused_scores = (
            self._reciprocal_rank_fusion([
                bm25_ranking,
                semantic_ranking,
            ])
        )

        candidates = []

        for index in fused_ranking:
            document = self.documents[index]

            candidates.append({
                "id": document["id"],
                "source": document["source"],
                "text": document["text"],
                "score": float(fused_scores[index]),
            })

        reranked = self.reranker.rerank(
            query,
            candidates,
        )

        return reranked[offset:offset + top_k]
