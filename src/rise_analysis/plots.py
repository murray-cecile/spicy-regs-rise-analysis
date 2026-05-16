"""Plots for exploratory analysis of comment metadata."""

from __future__ import annotations

from typing import Any

import altair as alt
import polars as pl

DEFAULT_DATE_COLUMN = "receive_date"


def _comments_to_frame(comments: list[dict[str, Any]] | pl.DataFrame) -> pl.DataFrame:
    """Normalize input to a Polars DataFrame."""
    if isinstance(comments, pl.DataFrame):
        return comments.clone()
    return pl.DataFrame(comments)


def plot_comment_volume_over_time(
    comments: list[dict[str, Any]] | pl.DataFrame,
    *,
    date_column: str = DEFAULT_DATE_COLUMN,
    width: int = 640,
    height: int = 240,
    title: str = "Comment volume by receive date",
) -> alt.Chart:
    """Altair line chart of comment counts aggregated by calendar day (UTC).

    ``date_column`` values are parsed as UTC, bucketed to the date in UTC, then
    counted. Suitable for ``receive_date`` ISO strings from the JSON export.

    Args:
        comments: Comment records or a Polars DataFrame that includes ``date_column``.
        date_column: ISO-8601 datetime column (e.g. ``receive_date``).
        width: Chart width in pixels.
        height: Chart height in pixels.
        title: Chart title text.

    Returns:
        An Altair chart (display in Jupyter with the last expression or ``.display()``).

    Raises:
        KeyError: If ``date_column`` is missing from the data.
    """
    df = _comments_to_frame(comments)
    if date_column not in df.columns:
        raise KeyError(f"Missing column {date_column!r}; columns: {df.columns}")

    daily = (
        df.select(
            pl.col(date_column).str.to_datetime(time_zone="UTC").dt.date().alias("_day"),
        )
        .group_by("_day")
        .agg(pl.len().alias("n_comments"))
        .sort("_day")
    )

    return (
        alt.Chart(daily)
        .mark_line(point=True)
        .encode(
            x=alt.X("_day:T", title="Date (UTC)"),
            y=alt.Y("n_comments:Q", title="Comment count"),
        )
        .properties(title=title, width=width, height=height)
    )
