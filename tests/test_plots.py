"""Tests for ``rise_analysis.plots``."""

from __future__ import annotations

import altair as alt
import polars as pl
import pytest

from rise_analysis.plots import plot_comment_volume_over_time


def test_plot_comment_volume_over_time_smoke() -> None:
    """Chart builds and encodes daily counts from ISO receive_date strings."""
    comments = [
        {"comment_id": "1", "receive_date": "2026-01-30T05:00:00Z"},
        {"comment_id": "2", "receive_date": "2026-01-30T12:00:00Z"},
        {"comment_id": "3", "receive_date": "2026-01-31T05:00:00Z"},
        {"comment_id": "4", "receive_date": "2026-01-31T06:00:00Z"},
        {"comment_id": "5", "receive_date": "2026-02-01T05:00:00Z"},
    ]
    chart = plot_comment_volume_over_time(comments)
    assert isinstance(chart, alt.Chart)
    spec = chart.to_dict()
    assert spec["mark"] == {"type": "line", "point": True}
    assert spec["encoding"]["y"]["field"] == "n_comments"

    df = pl.DataFrame(comments)
    chart_df = plot_comment_volume_over_time(df)
    assert isinstance(chart_df, alt.Chart)


def test_plot_comment_volume_over_time_missing_column_raises() -> None:
    """Missing date column raises KeyError."""
    with pytest.raises(KeyError, match="receive_date"):
        plot_comment_volume_over_time([{"comment_id": "x"}])
