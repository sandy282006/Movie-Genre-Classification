# Movie Genre Classification

## Objective
Classify movies into genres based on textual descriptions of their plots using machine learning.

## Features
- Text preprocessing using NLTK for cleaning and tokenization.
- TF-IDF vectorization for feature extraction.
- Random Forest classifier with hyperparameter tuning for genre prediction.

## Steps to Run
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the scripts in the following order:
   - `data_preprocessing.py`
   - `feature_extraction.py`
   - `model_training.py`
   - `evaluation.py`

## Dataset
The dataset consists of movie plots and their respective genres.

## Evaluation
Model performance is evaluated using accuracy, precision, recall, and F1-score.