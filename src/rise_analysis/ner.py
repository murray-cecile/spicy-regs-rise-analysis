"""Thin spaCy wrapper for organization named-entity recognition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import spacy

if TYPE_CHECKING:
    from spacy.language import Language

DEFAULT_MODEL = "en_core_web_md"


def load_nlp(model: str = DEFAULT_MODEL) -> Language:
    """Load a spaCy pipeline for English NER.

    Tries ``spacy.load`` first (works when the model is registered, e.g. after
    ``uv sync`` with ``en-core-web-md`` in project dependencies). Falls back to
    ``import en_core_web_md`` when the package is installed as a wheel only.

    Args:
        model: Pipeline name (default ``en_core_web_md``).

    Returns:
        Loaded spaCy ``Language`` object.

    Raises:
        OSError: If the model cannot be found or loaded.
    """
    try:
        return spacy.load(model)
    except OSError:
        if model != DEFAULT_MODEL:
            raise
    import en_core_web_md

    return en_core_web_md.load()


def extract_org_entities(text: str, nlp: Language) -> list[str]:
    """Return organization entity texts from a single comment body.

    Args:
        text: Plain text (typically output of :func:`rise_analysis.text_cleaning.clean_comment_text`).
        nlp: Loaded spaCy pipeline from :func:`load_nlp`.

    Returns:
        Organization strings in document order, deduplicated (first occurrence kept).
    """
    doc = nlp(text)
    seen: set[str] = set()
    orgs: list[str] = []
    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        normalized = ent.text.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        orgs.append(normalized)
    return orgs
