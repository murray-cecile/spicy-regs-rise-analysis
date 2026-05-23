"""Tests for ``rise_analysis.topics``."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rise_analysis.topics import ensure_nltk_stopwords, fit_topics, top_terms_per_topic

_SAMPLE_STOPWORDS = ["the", "a", "and"]


@pytest.fixture
def synthetic_corpus() -> list[str]:
    """Repeated comment-like texts for LDA tests (20 documents, 5 themes)."""
    return [
        "graduate loan limits professional programs mental health counseling rural communities",
        "graduate loan limits professional programs mental health counseling rural communities support",
        "income contingent repayment plans sunset borrowers rehabilitation second chance",
        "income contingent repayment plans sunset borrowers rehabilitation second chance default",
        "higher education act one big beautiful bill act loan caps department education",
        "higher education act one big beautiful bill act loan caps department education rule",
        "nursing MSN DNP professional degree definition exclude post baccalaureate programs",
        "nursing MSN DNP professional degree definition exclude post baccalaureate programs students",
        "public service loan forgiveness private loans graduate students financing demands",
        "public service loan forgiveness private loans graduate students financing demands research",
    ] * 2


@patch("rise_analysis.topics.ensure_nltk_stopwords", return_value=_SAMPLE_STOPWORDS)
def test_fit_topics_returns_models(_mock_stopwords: object, synthetic_corpus: list[str]) -> None:
    """LDA and vectorizer fit on a small repeated corpus."""
    lda, vectorizer = fit_topics(synthetic_corpus, n_topics=3, max_features=200)
    assert lda.n_components == 3
    assert vectorizer.get_feature_names_out().size > 0


@patch("rise_analysis.topics.ensure_nltk_stopwords", return_value=_SAMPLE_STOPWORDS)
def test_top_terms_per_topic_shape(_mock_stopwords: object, synthetic_corpus: list[str]) -> None:
    """Each topic returns the requested number of terms."""
    lda, vectorizer = fit_topics(synthetic_corpus, n_topics=2, max_features=150)
    topics = top_terms_per_topic(lda, vectorizer, n_terms=5)
    assert len(topics) == 2
    assert all(len(t) == 5 for t in topics)


@patch("rise_analysis.topics.ensure_nltk_stopwords", return_value=_SAMPLE_STOPWORDS)
def test_fit_topics_empty_raises(_mock_stopwords: object) -> None:
    """All-empty input is rejected."""
    with pytest.raises(ValueError, match="non-empty"):
        fit_topics(["", "   "])


@pytest.mark.slow
def test_ensure_nltk_stopwords_returns_list() -> None:
    """Stopwords load from NLTK; requires NLTK_DATA to be set and nlp-setup run."""
    words = ensure_nltk_stopwords()
    assert isinstance(words, list)
    assert len(words) > 10
    assert "the" in words


def test_ensure_nltk_stopwords_raises_without_corpus() -> None:
    """RuntimeError with helpful message when corpus is missing."""
    with patch("nltk.corpus.stopwords.words", side_effect=LookupError("not found")):
        with pytest.raises(RuntimeError, match="task nlp-setup"):
            ensure_nltk_stopwords()
