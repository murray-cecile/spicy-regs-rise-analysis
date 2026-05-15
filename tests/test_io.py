"""Tests for ``rise_analysis.io``."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from rise_analysis.io import load_comments_json


def test_load_comments_json_array() -> None:
    """Flat JSON array of comments loads as a list."""
    records = [
        {"comment_id": "1", "receive_date": "2026-01-01T05:00:00Z"},
        {"comment_id": "2", "receive_date": "2026-01-02T05:00:00Z"},
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "comments.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        assert load_comments_json(path) == records


def test_load_comments_json_wrapped() -> None:
    """Full dump with ``comments`` key returns inner list."""
    payload = {
        "docket": {},
        "comments": [{"comment_id": "a", "receive_date": "2026-03-01T05:00:00Z"}],
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "full.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        out = load_comments_json(path)
        assert len(out) == 1 and out[0]["comment_id"] == "a"


def test_load_comments_json_invalid_structure() -> None:
    """Non-list, non-wrapped payload raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bad.json"
        path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
        with pytest.raises(ValueError, match="Expected a JSON array"):
            load_comments_json(path)
