"""RISE comment analysis package."""

from rise_analysis.duplicates import assign_duplicate_clusters
from rise_analysis.io import load_comments_json
from rise_analysis.plots import plot_comment_volume_over_time
from rise_analysis.text_cleaning import (
    CleanedComment,
    clean_comment_text,
    remove_boilerplate_phrases,
    strip_html_to_text,
)

__all__ = [
    "CleanedComment",
    "assign_duplicate_clusters",
    "clean_comment_text",
    "load_comments_json",
    "plot_comment_volume_over_time",
    "remove_boilerplate_phrases",
    "strip_html_to_text",
]
