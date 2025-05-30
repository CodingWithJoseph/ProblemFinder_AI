# Phase 1: EDA & FE 
Note: Current values represent training data

## Exploratory Data Analysis (EDA) Checklist

### Basic Dataset Analysis
- [X] Check dataset dimensions (rows/columns)
  - 498 Examples 
  - 28 Features
- [X] Examine data types and verify consistency
  - bool (2) 
  - float64 (5)
  - int64 (12) 
  - object (9)
- [X] Identify and handle missing values

### Text Statistics Analysis
- [X] Calculate title length distribution (characters and words) 
  - **Titles show similar statistics** for both “Viable” and “Not Viable.”
      - Means and medians are almost the same.
      - Suggests that, regardless of the eventual viability, users put similar effort or structure into crafting their titles

- [X] Calculate body length distribution (characters and words)
  - **Not Viable:**
    - Much higher max, mean, and median.
    - Example: Max word count (6637 vs 3153), mean (551 vs 436), median (261 vs 266).
    - **Interpretation:** Some "Not Viable" posts go into extensive detail—possibly over-explaining or including lots of extraneous information.

  - **Viable:**
      - More concise, rarely as lengthy as the extremes in "Not Viable."
      - **Interpretation:** "Viable" posts may be more focused, to the point, and easier to assess quickly, indicating focus and clarity.

- [X] Generate summary statistics (min, max, mean, median)

### Question Pattern Analysis
- [ ] Count question marks per post
- [ ] Identify posts starting with question words
- [ ] Calculate question-to-sentence ratio
- [ ] Extract common question phrases
- [ ] Create boolean feature for posts starting with questions
- [ ] Identify common question patterns with regex
- [ ] Calculate question-to-text ratio

### Visualization Tasks
- [ ] Create histograms of text length metrics
- [ ] Generate word clouds from titles and bodies
- [ ] Plot subreddit distribution bar chart
- [ ] Create a correlation heatmap of numeric features

## 2. Feature Engineering Steps

### Text Length Features
- [ ] Add `title_char_count` and `title_word_count` columns
- [ ] Add `body_char_count` and `body_word_count` columns
- [ ] Calculate `text_density` (words/characters ratio)
- [ ] Determine the number of paragraphs in the body

### Sentiment Analysis Options
- [ ] **Option 1: Use existing libraries**
  - [ ] Implement TextBlob for quick sentiment scoring
  - [ ] Use VADER from NLTK for social media-focused sentiment
  - [ ] Add `title_sentiment` and `body_sentiment` columns
  
- [ ] **Option 2: Train custom sentiment model**
  - [ ] Gather labeled sentiment data (optional)
  - [ ] Extract relevant features (word embeddings, n-grams)
  - [ ] Train simple sentiment classifier
  - [ ] Apply model to generate sentiment scores

### Problem-Indicating Keywords
- [ ] Create a list of problem-indicating phrases
```python
problem_phrases = [
    "need help", "looking for", "is there a", "wish there was",
    "can't find", "how do I", "struggle with", "any tool for",
    "is there any", "help me", "solution for"
  ]
```
- [ ] Count occurrences of each phrase
- [ ] Create aggregate `problem_phrase_count` feature
- [ ] Create boolean features for common phrases

### Additional Text Features
- [ ] Extract URLs mentioned (potential existing solutions)
- [ ] Count specific characters (%, $, numbers)
- [ ] Identify technical terminology frequency
- [ ] Calculate readability scores

## Tokenization:
- [ ] Apply tokenization (e.g., nltk.word_tokenize or spacy.tokenizer).