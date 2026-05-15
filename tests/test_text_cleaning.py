"""Tests for ``rise_analysis.text_cleaning``."""

from __future__ import annotations

import pytest

from rise_analysis.text_cleaning import (
    clean_comment_text,
    remove_boilerplate_phrases,
    strip_html_to_text,
)


def test_strip_html_none_returns_empty() -> None:
    """None input yields empty string."""
    assert strip_html_to_text(None) == ""


def test_strip_html_tags_and_entities() -> None:
    """Tags removed; curly quotes and ampersand entities normalized."""
    raw = "Hello <b>world</b> &ldquo;quote&rdquo; Lewis &amp; Clark <br/><br/>next &#8217;s apostrophe."
    out = strip_html_to_text(raw)
    assert "<" not in out and ">" not in out
    assert '"quote"' in out
    assert "Lewis & Clark" in out
    assert "’s apostrophe" in out or "'s apostrophe" in out


def test_strip_html_collapses_whitespace() -> None:
    """Whitespace inside and around tags collapses to single spaces."""
    assert strip_html_to_text("a   <br/>  b") == "a b"


def test_remove_boilerplate_attach_variants() -> None:
    """Several 'see attached' spellings are stripped."""
    cases = [
        "See attached file.",
        "see attached files",
        "SEE ATTACHED FILE(S)",
        "Prefix see attached file(s) suffix",
    ]
    for c in cases:
        cleaned, attach, _dock = remove_boilerplate_phrases(c)
        assert attach is True
        assert "attached" not in cleaned.lower()


def test_remove_boilerplate_docket_comment() -> None:
    """Comment ED-2025-OPE-0944 phrase is removed."""
    text = "Please consider this. Comment ED-2025-OPE-0944 Thanks."
    cleaned, attach, dock = remove_boilerplate_phrases(text)
    assert dock is True
    assert attach is False
    assert "ED-2025-OPE-0944" not in cleaned


def test_remove_boilerplate_no_match() -> None:
    """Unrelated text passes through with flags false."""
    text = "We support the proposed rule on graduate lending."
    cleaned, attach, dock = remove_boilerplate_phrases(text)
    assert cleaned == text
    assert attach is False
    assert dock is False


def test_clean_comment_text_pipeline_and_trivial() -> None:
    """Full pipeline sets trivial when only boilerplate remains."""
    raw_html = "See attached file(s).<br/>Comment ED-2025-OPE-0944"
    result = clean_comment_text(raw_html, trivial_max_chars=15)
    assert result.removed_attach_boilerplate is True
    assert result.removed_docket_boilerplate is True
    assert result.is_trivial_after_clean is True


def test_clean_comment_text_non_trivial() -> None:
    """Long substantive comment is not flagged trivial with default threshold."""
    raw = "<p>Hello &amp; thanks for reading this substantive public comment.</p>"
    result = clean_comment_text(raw)
    assert result.is_trivial_after_clean is False
    assert "substance" in result.text or "substantive" in result.text


@pytest.mark.parametrize(
    ("length", "expected_trivial"),
    [(14, True), (15, False)],
)
def test_trivial_threshold_boundary(length: int, expected_trivial: bool) -> None:
    """Trivial flag uses ``len(text) < trivial_max_chars`` (default 15)."""
    text = "x" * length
    result = clean_comment_text(text, trivial_max_chars=15)
    assert result.is_trivial_after_clean is expected_trivial
