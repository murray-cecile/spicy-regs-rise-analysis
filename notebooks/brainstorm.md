# Plan

## Deliverables

Categorize commenters into stakeholder types by extracting self-identified organizations from comment text and detecting form letter campaigns.

A short narrative summarizing:

Who are the most active organizational commenters?
What are the major form letter campaigns and who appears to be behind them?
What's the ratio of organizational vs. individual comments?
Any notable patterns in submission timing or comment volume?

## Preprocessing

- extract the comment data... I need receive_date and comment. 
- get rid of all the see attached file variations, and the "Comment on ED-2025-OPE-0944"
- clean html and strip tags
- apply spacy

## Outstanding questions

- is the title of the comment useful?
- can I use an LLM or something else to code comments by type of commenter: institutional, an individual professional, etc
- 
