"""Tests for ``rise_analysis.duplicates``."""

from __future__ import annotations

import polars as pl
import pytest

from rise_analysis.duplicates import (
    _UnionFind,
    _apply_exact_matches,
    _apply_fuzzy_block,
    _build_length_blocks,
    _normalize_for_exact_match,
    assign_duplicate_clusters,
)


# ---------------------------------------------------------------------------
# _UnionFind
# ---------------------------------------------------------------------------


def test_union_find_merges_pairs() -> None:
    """Union-find links transitive sets."""
    uf = _UnionFind(4)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(3)


def test_union_find_idempotent() -> None:
    """Unioning the same pair twice does not corrupt state."""
    uf = _UnionFind(2)
    uf.union(0, 1)
    uf.union(0, 1)
    assert uf.find(0) == uf.find(1)


# ---------------------------------------------------------------------------
# _normalize_for_exact_match
# ---------------------------------------------------------------------------


def test_normalize_for_exact_match() -> None:
    """Exact-match key is lowercased, whitespace-collapsed."""
    assert _normalize_for_exact_match("  Foo\tBAR  ") == "foo bar"


# ---------------------------------------------------------------------------
# _apply_exact_matches
# ---------------------------------------------------------------------------


def test_apply_exact_matches_unions_shared_keys() -> None:
    """Indices with the same normalized key land in one cluster."""
    uf = _UnionFind(3)
    _apply_exact_matches(uf, ["hello world", "hello world", "different text"])
    assert uf.find(0) == uf.find(1)
    assert uf.find(0) != uf.find(2)


# ---------------------------------------------------------------------------
# _build_length_blocks
# ---------------------------------------------------------------------------


def test_build_length_blocks_groups_by_bucket() -> None:
    """Texts of similar length end up in the same bucket."""
    texts = ["a" * 50, "b" * 55, "c" * 200]
    blocks = _build_length_blocks(texts, bucket_size=80)
    # 50//80 == 0, 55//80 == 0 → same bucket; 200//80 == 2 → separate
    bucket_0 = blocks[0]
    assert 0 in bucket_0 and 1 in bucket_0
    assert 2 not in bucket_0


# ---------------------------------------------------------------------------
# _apply_fuzzy_block
# ---------------------------------------------------------------------------


def test_apply_fuzzy_block_merges_near_duplicate() -> None:
    """High-overlap texts within a block are merged by the helper directly."""
    shared = "Department proposed rule graduate professional loan limits " * 6
    texts = [shared + "suffix one.", shared + "suffix two."]
    uf = _UnionFind(2)
    _apply_fuzzy_block(uf, [0, 1], texts, threshold=0.80, max_features=512)
    assert uf.find(0) == uf.find(1)


def test_apply_fuzzy_block_skips_all_empty() -> None:
    """All-empty block does not raise and leaves union-find unchanged."""
    uf = _UnionFind(2)
    _apply_fuzzy_block(uf, [0, 1], ["", ""], threshold=0.9, max_features=512)
    assert uf.find(0) != uf.find(1)


# ---------------------------------------------------------------------------
# assign_duplicate_clusters (integration)
# ---------------------------------------------------------------------------


def test_assign_exact_duplicates_single_cluster() -> None:
    """Identical cleaned text ends in the same cluster."""
    body = "<p>The Department should reconsider the proposed caps.</p>"
    comments = [
        {"comment_id": "a", "comment": body},
        {"comment_id": "b", "comment": body},
        {"comment_id": "c", "comment": body},
    ]
    out = assign_duplicate_clusters(comments)
    assert out["cluster_id"].n_unique() == 1
    assert out.height == 3


def test_assign_exact_normalization_merges_whitespace_and_case() -> None:
    """Exact signature treats case and whitespace as equivalent."""
    comments = [
        {"comment_id": "1", "comment": "Hello   World"},
        {"comment_id": "2", "comment": "hello world"},
    ]
    out = assign_duplicate_clusters(comments)
    assert out["cluster_id"].n_unique() == 1


def test_assign_fuzzy_near_duplicates_merge() -> None:
    """Small tail-only edits keep high similarity within the same length bucket."""
    shared = (
        "We respectfully submit this comment regarding ED-2025-OPE-0944. "
        "Congress intended higher loan limits for professional programs. "
        "Students in counseling and education need predictable federal support. "
    ) * 3
    comments = [
        {"comment_id": "x", "comment": shared + "Respectfully submitted, Jane Doe, Example University."},
        {"comment_id": "y", "comment": shared + "Respectfully submitted, John Roe, Sample College."},
    ]
    out = assign_duplicate_clusters(comments, similarity_threshold=0.78)
    assert out["cluster_id"].n_unique() == 1


def test_assign_different_campaigns_stay_separate() -> None:
    """Unrelated long texts should not reach similarity threshold."""
    comments = [
        {"comment_id": "1", "comment": "Alpha campaign text about apples bananas cherries daily harvest festivals."},
        {"comment_id": "2", "comment": "Beta campaign text about zebras quotas tariffs yearly import volumes."},
    ]
    out = assign_duplicate_clusters(comments, similarity_threshold=0.95)
    assert out["cluster_id"].n_unique() == 2


def test_assign_polars_input() -> None:
    """Accepts a Polars DataFrame."""
    df = pl.DataFrame(
        {
            "comment_id": ["1", "2"],
            "comment": ["See attached.", "See attached."],
        }
    )
    out = assign_duplicate_clusters(df)
    assert out.height == 2
    assert out["cluster_id"].n_unique() == 1


def test_missing_columns_raise() -> None:
    """Required columns surface as KeyError."""
    with pytest.raises(KeyError, match="comment"):
        assign_duplicate_clusters([{"comment_id": "1"}])


def test_empty_comments_raise() -> None:
    """Empty input is rejected."""
    with pytest.raises(ValueError, match="non-empty"):
        assign_duplicate_clusters([])


def test_invalid_threshold_raises() -> None:
    """Similarity threshold must lie in the unit interval."""
    with pytest.raises(ValueError, match="similarity_threshold"):
        assign_duplicate_clusters(
            [{"comment_id": "1", "comment": "x"}],
            similarity_threshold=1.5,
        )
