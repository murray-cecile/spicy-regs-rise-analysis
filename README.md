## Reimagining and Improving Student Education (RISE) Analysis

This analysis takes public comment data extracted via the spicy-regs project and seeks to understand who commented and what comment campaigns appear to have occurred, as well as visualize the submission timing and comment volume.

### Summary

Analyze the 17,730 public comments on ED-2025-OPE-0944 — the Reimagining and Improving Student Education (RISE) proposed rule — to identify which organizations, coalitions, and interest groups are commenting.

### Background

This docket covers proposed amendments to Federal student loan programs under the Higher Education Act, implementing changes from the One Big Beautiful Bill Act. Key topics include redefining "graduate" vs "professional" program loan limits, simplifying repayment plans, sunsetting ICR plans, and loan rehabilitation provisions.

Comment period: Jan 30 – Mar 3, 2026
Total comments: 17,730
Data file: ~/Downloads/ED-2025-OPE-0944_2026-03-30.json (29.7 MB)
Problem

The regulations.gov data has no organization field — all 17,730 comments are typed as "Public Submission" with a generic title. We can't tell who is commenting without analyzing the comment text itself.

### Project Goals

#### Core questions

- Any notable patterns in submission timing or comment volume?
- What are the major form letter campaigns and who appears to be behind them?
- Who are the most active organizational commenters?
- What's the ratio of organizational vs. individual comments?

##### Stakeholder summary table

Categorize commenters into stakeholder types by extracting self-identified organizations from comment text and detecting form letter campaigns.

#### Work completed to date

- exploratory data analysis to understand data structure and nature of cleaning needed
- small Named Entity Recognition experiment to understand utility
- pre-processing methods implemented
- plot aggregate comment volume over the submission period
- LDA topic modeling over the comment text

### Findings

- Comment volume fluctuated over the week, with the most comments occuring midweek. 
- Comment volume spiked at the end of the comment period.
- Many commenters attached comments as PDFs or other files. These comments are not available for this analysis. This is probably a significant limitation on identifying organizations who commented and any messaging coordination between them, as organizations are probably more likely to submit formal letters or other documents.
- Among individual commenters, there is evidence of one or more form letter campaigns. In particular, based on the fuzzy-matching duplicates analysis, the American Nurses Association and other state nurses associations appear to have coordinated one such campaign.
- Major topic groupings in the response include references to nursing, social work, anesthesiology, and mental health. In addition - and unsurprisingly given the focus of the RISE proposed rule - the comments referenced professional and graduate degrees, loans, and access concerns.  

### Future work

- Additional Named Entity Recognition work to try to identify comment authors
- Classification of comments as coming from individuals vs. organizations

### Data Notes

- Comment fields available: comment_id, docket_id, agency_code, title, comment, document_type, posted_date, modify_date, receive_date
- Comments are HTML-encoded — decode before NLP processing
- No attachments are included in this dataset (attachment extraction is tracked in Add Attachment URLs to Comments in ETL Pipeline #10)

### Installation

Prerequisites:

- `uv` (`pipx install uv`) for Python dependency management
- `go-task` (`brew install go-task`) for utility commands

#### NLP setup

The English pipeline `en_core_web_md` is declared as a project dependency (wheel URL in `pyproject.toml`). After `uv sync`, use `from rise_analysis import load_nlp, extract_org_entities`.

NLTK stopwords for LDA live in `.venv/nltk_data/` (inside the venv, already gitignored). Run `task nlp-setup` once per clone to download them. Task sets `NLTK_DATA` automatically; for notebooks use the project `.venv` kernel (NLTK searches `.venv/nltk_data` by default) or export `NLTK_DATA=$PWD/.venv/nltk_data` before starting Jupyter.

To add another spaCy model, use `uv add` with the model's release wheel URL (see `[tool.uv.sources]` for `en-core-web-md`); `uv add <model_name>` alone usually will not resolve on PyPI.