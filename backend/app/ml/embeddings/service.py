from __future__ import annotations

import os
from typing import Iterable

import numpy as np


class EmbeddingService:
    """
    Lightweight embedding service for MediLens.

    Render's free instance has limited memory, so TF-IDF is used
    by default instead of loading SentenceTransformer/PyTorch.

    SentenceTransformer can still be enabled explicitly by setting:

        MEDILENS_USE_SENTENCE_TRANSFORMER=true
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "MEDILENS_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        self.use_sentence_transformer = (
            os.getenv(
                "MEDILENS_USE_SENTENCE_TRANSFORMER",
                "false",
            ).lower()
            in {"1", "true", "yes", "on"}
        )

        self._model = None
        self._vectorizer = None
        self._fitted = False

    def fit(
        self,
        corpus: Iterable[str],
    ) -> "EmbeddingService":
        corpus_list = list(corpus)

        # Only load SentenceTransformer when explicitly enabled.
        if (
            self.use_sentence_transformer
            and self._try_load_sentence_transformer()
        ):
            self._fitted = True
            return self

        # Lightweight fallback for Render.
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=4096,
        )

        self._vectorizer.fit(
            corpus_list or ["medical report"]
        )

        self._fitted = True
        return self

    def encode(
        self,
        texts: Iterable[str],
    ) -> np.ndarray:
        payload = list(texts)

        if not payload:
            return np.zeros(
                (0, 0),
                dtype="float32",
            )

        # SentenceTransformer is optional and disabled by default.
        if (
            self.use_sentence_transformer
            and self._try_load_sentence_transformer()
        ):
            vectors = self._model.encode(
                payload,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            return np.asarray(
                vectors,
                dtype="float32",
            )

        # Make sure TF-IDF has been fitted.
        if self._vectorizer is None:
            self.fit(payload)

        return (
            self._vectorizer
            .transform(payload)
            .toarray()
            .astype("float32")
        )

    def _try_load_sentence_transformer(self) -> bool:
        if self._model is not None:
            return True

        try:
            from sentence_transformers import (
                SentenceTransformer,
            )

            self._model = SentenceTransformer(
                self.model_name
            )

            return True

        except Exception:
            self._model = None
            return False
