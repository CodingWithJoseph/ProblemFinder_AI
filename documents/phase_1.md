# Phase 1: EDA & FE 
Note: Current values represent training data

## Exploratory Data Analysis (EDA) Checklist

### Basic Dataset Analysis
- [X] Check dataset dimensions (rows/columns)
  - (498, 26)
- [X] Examine data types and verify consistency
  - bool: 2
  - float64: 5 
  - int64: 12 
  - object: 7
- [X] Identify and handle missing values

### Text Statistics Analysis
- [x] Calculate title length distribution (characters and words)
- [x] Calculate body length distribution (characters and words)
- [X] Generate summary statistics (min, max, mean, median)
  - **Key Findings:**
    - **Market viable posts** (19.9% of dataset) show distinct text patterns
    - **Titles**: Slightly longer, more descriptive titles with fewer extremes
    - **Bodies**: More concise content with higher minimum substance
    - **Distribution**: Less variability, avoiding both extremely short and excessively long content
    - **Goldilocks Effect**: Viable posts occupy a "sweet spot" in length characteristics
  - Text Length Analysis: All Posts vs. Market Viable Posts

| Metric | All Posts | Market Viable | Change |
|--------|-----------|--------------|--------|
| **Sample Size** | 498 posts | 99 posts (19.9%) | |
| **Title Length** |  |  |  |
| Average | 54.4 chars / 9.4 words | 58.0 chars / 9.6 words | +6.6% / +2.1% |
| Range | 2-277 chars / 1-49 words | 10-134 chars / 1-25 words | Narrower range |
| **Body Length** |  |  |  |
| Average | 3316 chars / 529 words | 2829 chars / 437 words | -14.7% / -17.5% |
| Range | 14-39930 chars / 2-6637 words | 179-19744 chars / 34-3153 words | Narrower range |

*This pattern suggests content quality correlates with moderate, focused writing rather than extreme brevity or verbosity.*

### Question Pattern Analysis
- [ ] Count question marks per post
- [ ] Identify posts starting with question words
- [ ] Calculate question-to-sentence ratio
- [ ] Extract common question phrases

### Subreddit Analysis
- [X] Count posts per subreddit
  - All Posts (498)
    - **449 unique subreddits** with highly fragmented distribution
  - Market Viable Posts (99)
    - **96 unique subreddits** with extremely even distribution
- [X] Identify most common subreddits
    - All Data
      - **Long tail pattern**: 76.4% of subreddits appear just once
      - **Limited concentration**: Highest count is 6 posts (Advice subreddit)
    - Mark Viable
      - **Near-perfect dispersion**: 97% of subreddits contain only one viable post
      - **No dominant communities**: Only 3 subreddits have 2 viable posts each
- [ ] Compare text characteristics across subreddits

**Minor clustering in tech/business niches (SaaS, AnalyticsAutomation)**

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

### Question-Related Features
- [ ] Count question marks (`question_count`)
- [ ] Create boolean feature for posts starting with questions
- [ ] Identify common question patterns with regex
- [ ] Calculate question-to-text ratio

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
- [ ] Create list of problem-indicating phrases
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