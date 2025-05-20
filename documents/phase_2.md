# Phase 2: Data Preprocessing Pipeline

## Load Your Labeled Dataset:
- [ ] Load your Reddit post data (text and labels) into a Pandas DataFrame.

## Initial Text Cleaning Function:
- [ ] Create a function to apply to each Reddit post:
  - [ ] Convert to lowercase.
  - [ ] Remove URLs.
  - [ ] Remove Reddit-specific elements (e.g., u/username, r/subreddit, &amp;).
  - [ ] Remove punctuation.
  - [ ] Remove numbers (unless numerically significant for your domain).
  - [ ] Remove extra whitespace.

## Tokenization & Stop Word Removal:
- [ ] Apply tokenization (e.g., nltk.word_tokenize or spacy.tokenizer).
- [ ] Remove common English stop words (using nltk.corpus.stopwords or a custom list).

## Lemmatization:
- [ ] Apply lemmatization (using nltk.stem.WordNetLemmatizer or spacy.lemmatizer).