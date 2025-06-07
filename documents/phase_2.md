# Phase 1: EDA & FE 
Note: Current values represent training data

## Initial Text Cleaning Function:
- [X] Create a function to apply to each Reddit post:
  - [X] Convert to lowercase.
  - [X] Remove URLs.
  - [X] Remove Reddit-specific elements (e.g., u/username, r/subreddit, &amp;).
  - [X] Remove extra whitespace.

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

### Visualization Tasks
- [X] Create histograms of text length metrics
- [X] Generate word clouds from titles and bodies

## 2. Feature Engineering Steps

### Text Length Features
- [X] Add `title_char_count` and `title_word_count` columns
- [X] Add `body_char_count` and `body_word_count` columns

### Sentiment Analysis Options
- [X] Use VADER from NLTK for social media-focused sentiment
- [X] Add `title_sentiment` and `body_sentiment` columns