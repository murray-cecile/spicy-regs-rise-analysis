"""Normalize comment bodies: HTML stripping and boilerplate phrase removal."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape

# "See attached file", "see attached files", "See attached file(s).", etc.
_ATTACH_BOILERPLATE_PATTERN = re.compile(
    r"(?i)\bsee\s+attached(?:\s+file(?:s|\(s\)))?\s*\.?",
)

# e.g. "Comment ED-2025-OPE-0944" or same with trailing punctuation
_DOCKET_BOILERPLATE_PATTERN = re.compile(
    r"(?i)\bcomment\s+ED-2025-OPE-0944\b[\s.,;:!?]*",
)


@dataclass(frozen=True, slots=True)
class CleanedComment:
    """Result of cleaning one regulatory comment body.

    Attributes:
        text: Text after HTML decoding/tag removal and boilerplate stripping.
        removed_attach_boilerplate: True if any "see attached …" boilerplate was removed.
        removed_docket_boilerplate: True if a Comment ED-2025-OPE-0944 fragment was removed.
        is_trivial_after_clean: True if ``text`` is empty or shorter than the trivial threshold.
    """

    text: str
    removed_attach_boilerplate: bool
    removed_docket_boilerplate: bool
    is_trivial_after_clean: bool


def strip_html_to_text(text: str | None) -> str:
    """Strip HTML tags, normalize a few entities, and collapse whitespace.

    Mirrors the logic developed in ``NamedEntityRecognition.ipynb``:
    curly quote entities and ``&amp;`` are normalized before tag stripping,
    then remaining numeric/named entities are resolved via ``html.unescape``.

    Args:
        text: Raw comment HTML or plain text. ``None`` is treated as empty.

    Returns:
        Plain text with normalized whitespace.
    """
    if text is None:
        return ""

    # Specific named entities for curly quotes, then ampersand (including chained &amp;amp;).
    out = text.replace("&ldquo;", '"').replace("&rdquo;", '"')
    while "&amp;" in out:
        out = out.replace("&amp;", "&")

    out = re.sub(r"<[^>]+>", " ", out)
    out = unescape(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def remove_boilerplate_phrases(text: str) -> tuple[str, bool, bool]:
    """Remove attachment and docket-ID boilerplate substrings.

    Args:
        text: Plain text (typically output of :func:`strip_html_to_text`).

    Returns:
        Tuple of ``(cleaned_text, removed_attach, removed_docket)``.
    """
    out, n_attach = _ATTACH_BOILERPLATE_PATTERN.subn(" ", text)
    out, n_docket = _DOCKET_BOILERPLATE_PATTERN.subn(" ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out, n_attach > 0, n_docket > 0


def clean_comment_text(raw: str | None, *, trivial_max_chars: int = 15) -> CleanedComment:
    """Full cleaning pipeline: HTML, boilerplate removal, trivial-length flag.

    Does not drop rows; callers use flags (e.g. ``is_trivial_after_clean``) to filter.

    Args:
        raw: Raw ``comment`` field from JSON.
        trivial_max_chars: ``is_trivial_after_clean`` is True when
            ``len(text) <= trivial_max_chars`` (after cleaning), or text is empty.

    Returns:
        :class:`CleanedComment` with plain text and boolean flags.
    """
    stripped = strip_html_to_text(raw)
    cleaned, removed_attach, removed_docket = remove_boilerplate_phrases(stripped)
    trivial = len(cleaned) < trivial_max_chars
    return CleanedComment(
        text=cleaned,
        removed_attach_boilerplate=removed_attach,
        removed_docket_boilerplate=removed_docket,
        is_trivial_after_clean=trivial,
    )
