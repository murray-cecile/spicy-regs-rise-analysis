"""RISE comment analysis package."""

from rise_analysis.duplicates import assign_duplicate_clusters
from rise_analysis.io import load_comments_json
from rise_analysis.ner import DEFAULT_MODEL, extract_org_entities, load_nlp
from rise_analysis.plots import plot_comment_volume_over_time
from rise_analysis.text_cleaning import (
    CleanedComment,
    clean_comment_text,
    remove_boilerplate_phrases,
    strip_html_to_text,
)
from rise_analysis.topics import ensure_nltk_stopwords, fit_topics, top_terms_per_topic

__all__ = [
    "CleanedComment",
    "DEFAULT_MODEL",
    "assign_duplicate_clusters",
    "clean_comment_text",
    "ensure_nltk_stopwords",
    "extract_org_entities",
    "fit_topics",
    "load_comments_json",
    "load_nlp",
    "plot_comment_volume_over_time",
    "remove_boilerplate_phrases",
    "strip_html_to_text",
    "top_terms_per_topic",
]
