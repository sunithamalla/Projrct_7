# NLP Project: Disaster Tweet Classification

## Objective
Build a machine learning system that classifies tweets as:
- 0 = Non-disaster
- 1 = Real disaster

## Dataset
The supplied Excel dataset is included as:
`twitter_disaster (1).xlsx`

## Technologies
- Python
- Pandas / NumPy
- Matplotlib / Seaborn
- Scikit-learn
- TF-IDF
- Logistic Regression
- Decision Tree
- Random Forest
- Flask
- Pickle

## Project Steps
1. Data exploration
2. Duplicate and missing-value handling
3. Text cleaning
4. EDA and keyword analysis
5. TF-IDF feature extraction
6. Train/test split
7. Model training and comparison
8. Evaluation with Accuracy, Precision, Recall, F1 and ROC-AUC
9. Confusion matrix, ROC and Precision-Recall curves
10. Save the best model
11. Flask web deployment

## How to run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the models
```bash
python disaster_tweet_classification.py
```

This creates:
- `disaster_tweet_model.pkl`
- `tfidf_vectorizer.pkl`
- files inside `outputs/`

### Start the web application
```bash
python app.py
```

Open the local Flask address shown in the terminal, normally:
`http://127.0.0.1:5000`

## Important
The model is a machine-learning classifier for the supplied dataset. A prediction should not be treated as confirmation of an actual emergency without human or authoritative verification.
