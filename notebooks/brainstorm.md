# Plan

## Preprocessing

- extract the comment data... I need receive_date and comment. 
- get rid of all the see attached file variations, and the "Comment on ED-2025-OPE-0944"
- clean html and strip tags
- duplicates: naive text matching and also tf-idf approach
- apply spacy

## Outstanding questions

- is the title of the comment useful?
- what can I use to code comments by type of commenter: institutional, an individual professional, etc
