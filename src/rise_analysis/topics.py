"""LDA topic modeling for cleaned comment text."""

from __future__ import annotations

from collections.abc import Sequence

import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

DEFAULT_N_TOPICS = 8
DEFAULT_MAX_FEATURES = 4096
DEFAULT_N_TERMS = 12


def ensure_nltk_stopwords() -> list[str]:
    """Return English NLTK stopwords.

    Reads from wherever NLTK resolves data — in practice the directory
    named by the ``NLTK_DATA`` environment variable. Run ``task nlp-setup``
    once per clone to download the corpus and set the variable.

    Returns:
        List of stopword strings for ``CountVectorizer(stop_words=...)``.

    Raises:
        RuntimeError: If the stopwords corpus is not found; instructs the
            caller to run ``task nlp-setup``.
    """
    try:
        return nltk.corpus.stopwords.words("english")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK stopwords corpus not found. "
            "Run `task nlp-setup` to download into .venv/nltk_data/ "
            "(or set NLTK_DATA to that path for notebooks)."
        ) from exc


def fit_topics(
    texts: Sequence[str],
    n_topics: int = DEFAULT_N_TOPICS,
    *,
    max_features: int = DEFAULT_MAX_FEATURES,
    max_iter: int = 20,
    random_state: int = 42,
) -> tuple[LatentDirichletAllocation, CountVectorizer]:
    """Fit LDA on a corpus of comment texts.

    Args:
        texts: Iterable of plain-text comments (empty strings are dropped).
        n_topics: Number of latent topics.
        max_features: Vocabulary size cap for bag-of-words.
        max_iter: Maximum EM iterations for LDA.
        random_state: Reproducible topic assignment seed.

    Returns:
        Tuple of ``(fitted LDA model, fitted CountVectorizer)``.

    Raises:
        ValueError: If no non-empty texts remain after filtering.
        RuntimeError: If NLTK stopwords are not available (see :func:`ensure_nltk_stopwords`).
    """
    corpus = [t.strip() for t in texts if t and t.strip()]
    if not corpus:
        raise ValueError("texts must contain at least one non-empty string")

    stop_words = ensure_nltk_stopwords()
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    document_term = vectorizer.fit_transform(corpus)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        random_state=random_state,
        learning_method="online",
    )
    lda.fit(document_term)
    return lda, vectorizer


def top_terms_per_topic(
    lda: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    n_terms: int = DEFAULT_N_TERMS,
) -> list[list[str]]:
    """Top weighted terms for each LDA topic.

    Args:
        lda: Fitted model from :func:`fit_topics`.
        vectorizer: Fitted vectorizer paired with ``lda``.
        n_terms: Number of terms to return per topic.

    Returns:
        One list of term strings per topic (topic index order).
    """
    feature_names = vectorizer.get_feature_names_out()
    topics: list[list[str]] = []
    for topic_weights in lda.components_:
        top_indices = topic_weights.argsort()[-n_terms:][::-1]
        topics.append([feature_names[i] for i in top_indices])
    return topics
