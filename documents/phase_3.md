# Phase 3: Model Training & Evaluation

## Feature Preparation
- [X] **Generate BERT Sentence Embeddings**:
  - Utilize a pre-trained Transformer model (e.g., BERT, RoBERTa) via Hugging Face's `transformers` library.
  - Apply the model's tokenizer to your `title` and `body` text columns.
  - Extract the sentence embeddings (e.g., the `[CLS]` token embedding) for both `title` and `body`.
  - Store these embeddings as new columns in your DataFrame (e.g., `title_bert`, `body_bert`).

## Split Data:
- [ ] Split your labeled dataset (now with BERT embeddings) into training, validation, and test sets (e.g., 70% train, 15% validation, 15% test) using `sklearn.model_selection.train_test_split`.
- [ ] Ensure the split is stratified to handle potential class imbalance (`stratify=y`).

## Select and Train Your Model:
- [ ] **Train a Classifier on BERT Embeddings**:
  - Choose a suitable classifier for your tabular data that incorporates the BERT embeddings as features. Options include:
    - **Logistic Regression** (good baseline, interpretable)
    - **Support Vector Classifier (SVC)** (often strong with dense features)
    - **RandomForestClassifier** or **GradientBoostingClassifier (e.g., LightGBM, XGBoost)** (powerful for tabular data)
    - **A small Neural Network** (e.g., using TensorFlow/Keras or PyTorch) for more complex interactions between embeddings and other features.
  - Train the chosen model on your combined feature set (`X`) and `market_viability` target (`y`).

## Evaluate Model Performance:
- [ ] Make predictions on your held-out test s et.
- [ ] Calculate key metrics:
  - [ ] **F1-score** (macro or weighted): Primary metric for binary classification with potential imbalance.
  - [ ] **Precision and Recall**: For each class (Viable Problem, Non-Viable Problem).
  - [ ] **Confusion Matrix**: Visualize where the model is making errors.