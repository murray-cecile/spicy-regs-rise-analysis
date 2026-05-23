"""Tests for ``rise_analysis.ner``."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rise_analysis.ner import extract_org_entities, load_nlp


def _make_ent(text: str, label: str) -> MagicMock:
    ent = MagicMock()
    ent.text = text
    ent.label_ = label
    return ent


def test_extract_org_entities_filters_and_dedupes() -> None:
    """Only ORG labels are returned; duplicates are dropped."""
    nlp = MagicMock()
    doc = MagicMock()
    doc.ents = [
        _make_ent("Lewis & Clark College", "ORG"),
        _make_ent("Department of Education", "ORG"),
        _make_ent("graduate", "MISC"),
        _make_ent("Lewis & Clark College", "ORG"),
    ]
    nlp.return_value = doc

    orgs = extract_org_entities("ignored", nlp)
    assert orgs == ["Lewis & Clark College", "Department of Education"]
    nlp.assert_called_once_with("ignored")


def test_extract_org_entities_skips_blank() -> None:
    """Whitespace-only entity text is omitted."""
    nlp = MagicMock()
    doc = MagicMock()
    doc.ents = [_make_ent("  ", "ORG"), _make_ent("Real Org", "ORG")]
    nlp.return_value = doc

    assert extract_org_entities("text", nlp) == ["Real Org"]


@pytest.mark.slow
def test_load_nlp_integration() -> None:
    """Load the project-pinned en_core_web_md model (requires uv sync)."""
    nlp = load_nlp()
    orgs = extract_org_entities(
        "Lewis & Clark College urges the Department of Education to reconsider.",
        nlp,
    )
    assert any("Lewis" in o or "Clark" in o for o in orgs)
