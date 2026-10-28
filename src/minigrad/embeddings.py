"""Embedding backends and a cached in-memory embedding index.

The pretrained backend is optional. Importing this module never imports
``sentence_transformers``; the dependency is loaded only when the model is
first used. That keeps the existing tests and experiments free of the
optional dependency, and unit tests can use :class:`HashingEmbeddingBackend`
or their own fake backend.

Pretrained backend
------------------
:class:`SentenceTransformerEmbeddings` wraps
``sentence-transformers/all-MiniLM-L6-v2`` (Apache-2.0), a small 384-dimension
sentence-embedding model. The recommended setting for cosine retrieval is to
L2-normalise the encoded vectors, so this backend does that by default. The
same backend instance must be used for passages and queries.

Caching
-------
:class:`EmbeddingIndex` embeds every chunk once, when the index is built, and
keeps the vectors in memory. Queries are embedded lazily and cached by their
exact text, so repeated evaluation passes re-use the query vectors. No model
files are downloaded by the code in this module; only the optional backend
can do that, on first use.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from minigrad.bm25 import tokenize
from minigrad.vector_search import rank_by_similarity

__all__ = [
    "EmbeddingModelUnavailable",
    "HashingEmbeddingBackend",
    "SentenceTransformerEmbeddings",
    "EmbeddingIndex",
    "DEFAULT_MODEL_NAME",
]

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModelUnavailable(RuntimeError):
    """Raised when the optional pretrained embedding model cannot be used."""


class HashingEmbeddingBackend:
    """A deterministic hashed bag-of-words vector for offline smoke tests.

    This is **not** a semantic embedding model. It hashes each token into one
    of ``dimension`` coordinates with a signed contribution, then L2-normalises
    the vector. Cosine similarity between two such vectors reflects shared
    surface tokens (and hash collisions), just like a lexical representation,
    so it cannot be expected to bridge "world" to "planet" or "natural
    satellites" to "moons". It exists so the indexing and ranking code can be
    exercised without a network connection and so the experiment has a
    reproducible fallback when the pretrained model cannot be fetched.

    The index and sign are taken from different bytes of the same BLAKE2b
    digest. Using separate bytes matters: deriving both from one integer makes
    the sign a function of the coordinate index when the dimension is even, so
    tokens sharing a coordinate could never cancel each other.
    """

    def __init__(self, dimension: int = 256):
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension
        self.model_name = f"hashing-{dimension}"
        self.normalize_embeddings = True
        self.was_downloaded = False
        self.resolved_model_name = self.model_name

    def _embed(self, text):
        vector = [0.0] * self.dimension
        tokens = tokenize(text) or ["<empty>"]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        length = math.sqrt(sum(value * value for value in vector))
        if length == 0.0:
            vector[0] = 1.0
            length = 1.0
        return [value / length for value in vector]

    def encode(self, texts, batch_size: int = 32):
        """Embed ``texts``; ``batch_size`` is accepted for API compatibility."""
        return [self._embed(text) for text in texts]


class SentenceTransformerEmbeddings:
    """Lazy wrapper around a pretrained Sentence Transformers model.

    ``model_name`` may be a Hub identifier (for example
    ``sentence-transformers/all-MiniLM-L6-v2``) or a local directory. The
    model is loaded only on the first ``encode`` call. By default it is loaded
    from the local path or Hugging Face cache when possible, and downloaded
    only if it is missing; set ``local_files_only=True`` to forbid downloads
    entirely. After loading, ``was_downloaded`` reports which case applied and
    ``resolved_model_name`` records the identifier or path that was used.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        local_files_only: bool = False,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.local_files_only = local_files_only
        self._model = None
        # None until the model has actually been loaded.
        self.was_downloaded = None
        self.resolved_model_name = None

    def load(self):
        """Load the model on first use and return it."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingModelUnavailable(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers`, or run the experiment "
                "with `--backend hash` for an offline, non-semantic fallback."
            ) from error

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device

        local_path = Path(self.model_name).expanduser()
        if local_path.exists():
            resolved = str(local_path)
            self._model = self._load_or_error(SentenceTransformer, resolved, kwargs)
            self.was_downloaded = False
            self.resolved_model_name = str(local_path.resolve())
            return self._model

        # Try the cache first so a locally available model never touches the
        # network and so the experiment can report whether a download occurred.
        try:
            self._model = self._load_or_error(
                SentenceTransformer,
                self.model_name,
                {**kwargs, "local_files_only": True},
            )
            self.was_downloaded = False
        except EmbeddingModelUnavailable:
            if self.local_files_only:
                raise
            self._model = self._load_or_error(SentenceTransformer, self.model_name, kwargs)
            self.was_downloaded = True

        self.resolved_model_name = self.model_name
        return self._model

    def _load_or_error(self, model_class, name, kwargs):
        try:
            return model_class(name, **kwargs)
        except Exception as error:
            where = "local cache" if kwargs.get("local_files_only") else (
                self.device or "the default device"
            )
            raise EmbeddingModelUnavailable(
                f"could not load embedding model {name!r} from {where}: {error}. "
                "Provide a local model path, put the model in the Hugging Face "
                "cache, or allow a download."
            ) from error

    def encode(self, texts, batch_size: int | None = None):
        """Embed texts with the model's recommended normalisation setting."""
        texts = list(texts)
        if not texts:
            return []
        model = self.load()
        batch_size = batch_size or self.batch_size
        try:
            encoded = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise EmbeddingModelUnavailable(
                f"embedding inference failed on {self.device or 'the default device'}: {error}"
            ) from error
        return [list(map(float, row)) for row in encoded]


class EmbeddingIndex:
    """An in-memory embedding cache with cosine-similarity search."""

    def __init__(self, chunks, backend, batch_size: int = 32):
        self.chunks = list(chunks)
        self.backend = backend
        self.batch_size = batch_size
        self.texts = [chunk.text for chunk in self.chunks]
        self.vectors = (
            [list(vector) for vector in backend.encode(self.texts, batch_size=batch_size)]
            if self.texts
            else []
        )
        self.query_cache = {}

        if self.vectors:
            self.dimension = len(self.vectors[0])
            if self.dimension == 0:
                raise ValueError("embedding backend returned an empty vector")
            for index, vector in enumerate(self.vectors):
                if len(vector) != self.dimension:
                    raise ValueError(
                        f"embedding dimension mismatch at chunk {index}: "
                        f"{len(vector)} != {self.dimension}"
                    )
                if not all(math.isfinite(value) for value in vector):
                    raise ValueError(
                        f"embedding at chunk {index} contains a non-finite value"
                    )
        else:
            self.dimension = 0

    def embed_query(self, query: str):
        """Return a cached query embedding.

        The query vector is validated against the corpus dimension and for
        finite values before it is cached or ranked. This keeps a broken
        backend from surfacing as a confusing cosine error later.
        """
        if query not in self.query_cache:
            vectors = self.backend.encode([query], batch_size=1)
            if len(vectors) != 1:
                raise ValueError("embedding backend must return one vector per query")
            vector = list(vectors[0])
            if len(vector) != self.dimension:
                raise ValueError(
                    f"query embedding dimension mismatch: {len(vector)} != "
                    f"{self.dimension}"
                )
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("query embedding contains a non-finite value")
            self.query_cache[query] = vector
        return self.query_cache[query]

    def search(self, query: str, top_k: int = 5, min_score=None):
        """Return ``(chunk_index, cosine_score)`` pairs, best first."""
        if not self.vectors:
            return []
        return rank_by_similarity(
            self.embed_query(query),
            self.vectors,
            top_k=top_k,
            min_score=min_score,
        )
