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

### Findings



### Data Notes

- Comment fields available: comment_id, docket_id, agency_code, title, comment, document_type, posted_date, modify_date, receive_date
- Comments are HTML-encoded — decode before NLP processing
- No attachments are included in this dataset (attachment extraction is tracked in Add Attachment URLs to Comments in ETL Pipeline #10)

### Installation

Prerequisites:

- `uv` for Python dependency management
- `go-task` (`brew install go-task`) for utility commands

#### NLP setup

`uv` doesn't play with spacy models as easily as pip. The English pipeline `en_core_web_md` is already declared as a project dependency, so no separate `python -m spacy download` step is required unless using a different corpus. To install alternative spacy models, use the direct URL dependency with `uv add`.