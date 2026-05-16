"""Near-duplicate and template detection for cleaned comment text."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from rise_analysis.text_cleaning import CleanedComment, clean_comment_text

DEFAULT_TEXT_COLUMN = "comment"
DEFAULT_ID_COLUMN = "comment_id"


class _UnionFind:
    """Disjoint-set union for merging duplicate indices."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        """Find root of set containing ``x`` with path compression."""
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, a: int, b: int) -> None:
        """Merge sets containing ``a`` and ``b``."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


def _normalize_for_exact_match(cleaned: str) -> str:
    """Lowercase and collapse whitespace for exact-signature matching."""
    return " ".join(cleaned.lower().split())


def _comments_to_frame(comments: list[dict[str, Any]] | pl.DataFrame) -> pl.DataFrame:
    """Utility function to convert comments to a Polars DataFrame."""
    if isinstance(comments, pl.DataFrame):
        return comments.clone()
    return pl.DataFrame(comments)


def _clean_texts(
    rows: list[dict[str, Any]],
    text_column: str,
    trivial_max_chars: int,
) -> list[CleanedComment]:
    """Apply ``clean_comment_text`` to each row and return cleaned objects."""
    return [clean_comment_text(row[text_column], trivial_max_chars=trivial_max_chars) for row in rows]


def _apply_exact_matches(uf: _UnionFind, exact_keys: list[str]) -> None:
    """Union all indices that share an exact normalized key."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(exact_keys):
        buckets[key].append(idx)
    for indices in buckets.values():
        first = indices[0]
        for other in indices[1:]:
            uf.union(first, other)


def _build_length_blocks(
    cleaned_texts: list[str],
    bucket_size: int,
) -> dict[int, list[int]]:
    """Group indices by length bucket for fuzzy comparison blocking.

    Only indices in the same bucket are compared pairwise, keeping
    the effective comparison set small for large corpora.
    """
    blocks: dict[int, list[int]] = defaultdict(list)
    for idx, text in enumerate(cleaned_texts):
        blocks[len(text) // max(bucket_size, 1)].append(idx)
    return blocks


def _apply_fuzzy_block(
    uf: _UnionFind,
    global_indices: list[int],
    block_texts: list[str],
    threshold: float,
    max_features: int,
) -> None:
    """Vectorize one length block, threshold with linear_kernel, and union pairs.

    Uses ``linear_kernel`` (dot product on L2-normalised TF-IDF vectors) which is
    equivalent to cosine similarity without the redundant normalization pass.
    Pair detection via ``np.argwhere`` on the upper triangle avoids a Python double loop.
    """
    if not any(block_texts):
        return
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(block_texts)
    except ValueError as exc:
        logger.debug("Skipping fuzzy block of size {}: {}", len(global_indices), exc)
        return

    sim = linear_kernel(matrix)
    # Extract upper triangle (k=1) to skip the diagonal and avoid counting pairs twice.
    pairs = np.argwhere(np.triu(sim, k=1) >= threshold)
    for bi, bj in pairs:
        uf.union(global_indices[bi], global_indices[bj])


def _build_result_frame(
    rows: list[dict[str, Any]],
    id_column: str,
    cluster_ids: list[int],
    cleaned_objs: list[CleanedComment],
) -> pl.DataFrame:
    """Assemble the output DataFrame from clustered and cleaned data."""
    cleaned_texts = [c.text for c in cleaned_objs]
    return pl.DataFrame(
        {
            id_column: [row[id_column] for row in rows],
            "cluster_id": cluster_ids,
            "text_clean": cleaned_texts,
            "is_trivial_after_clean": [c.is_trivial_after_clean for c in cleaned_objs],
            "removed_attach_boilerplate": [c.removed_attach_boilerplate for c in cleaned_objs],
            "removed_docket_boilerplate": [c.removed_docket_boilerplate for c in cleaned_objs],
        }
    )


def assign_duplicate_clusters(
    comments: list[dict[str, Any]] | pl.DataFrame,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
    similarity_threshold: float = 0.92,
    length_bucket_chars: int = 80,
    max_features: int = 4096,
    trivial_max_chars: int = 15,
) -> pl.DataFrame:
    """Cluster comments by exact text match and blocked fuzzy TF-IDF similarity.

    Each row is cleaned with :func:`rise_analysis.text_cleaning.clean_comment_text`
    (HTML + boilerplate removal) before deduplication. Index ``i`` is only compared
    fuzzily to ``j`` when ``len(text_i) // length_bucket_chars`` equals
    ``len(text_j) // length_bucket_chars``, keeping pairwise work small at ~17k scale.

    Args:
        comments: Records with ``id_column`` and raw ``text_column`` (HTML allowed).
        text_column: Raw comment body field name.
        id_column: Unique identifier field name.
        similarity_threshold: Cosine similarity in ``[0, 1]``; pairs at or above are merged.
        length_bucket_chars: Denominator for length blocking (smaller → larger blocks).
        max_features: ``TfidfVectorizer`` vocabulary cap per block.
        trivial_max_chars: Passed to :func:`clean_comment_text` for the trivial flag column.

    Returns:
        DataFrame with ``id_column``, ``cluster_id`` (dense int 0..K-1), ``text_clean``,
        ``is_trivial_after_clean``, ``removed_attach_boilerplate``,
        and ``removed_docket_boilerplate``.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If ``comments`` is empty or ``similarity_threshold`` not in ``[0, 1]``.
    """
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")

    df = _comments_to_frame(comments)
    if df.height == 0:
        raise ValueError("comments must be non-empty")
    if id_column not in df.columns or text_column not in df.columns:
        raise KeyError(
            f"Expected columns {id_column!r} and {text_column!r}; got {df.columns!r}",
        )

    rows = df.select(pl.col(id_column), pl.col(text_column)).to_dicts()
    cleaned_objs = _clean_texts(rows, text_column, trivial_max_chars)
    cleaned_texts = [c.text for c in cleaned_objs]

    uf = _UnionFind(df.height)
    _apply_exact_matches(uf, [_normalize_for_exact_match(t) for t in cleaned_texts])

    blocks = _build_length_blocks(cleaned_texts, length_bucket_chars)
    for indices in blocks.values():
        if len(indices) < 2:
            continue
        _apply_fuzzy_block(uf, indices, [cleaned_texts[i] for i in indices], similarity_threshold, max_features)

    roots = [uf.find(i) for i in range(df.height)]
    root_to_cluster = {r: k for k, r in enumerate(sorted(set(roots)))}
    cluster_ids = [root_to_cluster[r] for r in roots]

    return _build_result_frame(rows, id_column, cluster_ids, cleaned_objs)
