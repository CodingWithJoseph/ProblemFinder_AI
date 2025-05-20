# Phase 3: Model Training & Evaluation

## Choose Your Text Representation/Feature Engineering:
- [ ] **Recommended (High Performance)**: Use a pre-trained Transformer model (e.g., BERT, RoBERTa) via Hugging Face's transformers library.
  - [ ] Tokenize your cleaned text using the specific tokenizer for your chosen Transformer model (e.g., AutoTokenizer.from_pretrained('bert-base-uncased')).
  - [ ] Convert tokens to input IDs, attention masks, and token type IDs (as required by the model).
- [ ] **Good Baseline**: Implement TF-IDF Vectorization using sklearn.feature_extraction.text.TfidfVectorizer.
  - [ ] Fit and transform your preprocessed text.

## Split Data:
- [ ] Split your labeled dataset into training, validation, and test sets (e.g., 70% train, 15% validation, 15% test) using sklearn.model_selection.train_test_split.
- [ ] Ensure the split is stratified if your classes are imbalanced (stratify=y).

## Select and Train Your Model:
- [ ] **If using Transformers (Highly Recommended)**:
  - [ ] Load a pre-trained Transformer model with a sequence classification head (e.g., AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=your_num_labels)).
  - [ ] Set up a Trainer (from transformers.Trainer) or a custom training loop with an optimizer (e.g., AdamW) and a learning rate scheduler.
  - [ ] Train the model on your training data, evaluating performance on the validation set at regular intervals.
- [ ] **If using Traditional ML (Good Baseline/Quicker Iteration)**:
  - [ ] Choose a classifier: LogisticRegression, SVC (Support Vector Classifier), RandomForestClassifier.
  - [ ] Train the chosen model on your TF-IDF vectorized training data.

## Evaluate Model Performance:
- [ ] Make predictions on your held-out test set.
- [ ] Calculate key metrics:
  - [ ] F1-score (macro or weighted): Primary metric for multi-class classification, especially with imbalance.
  - [ ] Precision and Recall: For each class ("Pain Point," "Market Opportunity").
  - [ ] Confusion Matrix: Visualize where the model is making errors.