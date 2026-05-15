"""Load comment records from JSON exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def load_comments_json(path: str | Path) -> list[dict[str, Any]]:
    """Load comment dicts from a JSON file.

    Accepts either:

    - A JSON array of comment objects (``comments-only.json`` from prep), or
    - An object with a ``"comments"`` key containing that array (full dump).

    Args:
        path: Filesystem path to ``.json``.

    Returns:
        List of comment records (plain dicts).

    Raises:
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the structure is neither a list nor a dict with ``comments``.
    """
    raw_path = Path(path)
    with raw_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return cast(list[dict[str, Any]], payload)

    if isinstance(payload, dict) and "comments" in payload:
        comments = payload["comments"]
        if not isinstance(comments, list):
            raise ValueError('"comments" must be a JSON array')
        return cast(list[dict[str, Any]], comments)

    raise ValueError(
        f"Expected a JSON array of comments or an object with a 'comments' array; got {type(payload).__name__}",
    )
