# ============================================================
# DISASTER TWEET CLASSIFICATION USING NLP
# ============================================================
import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

FILE_PATH = "twitter_disaster (1).xlsx"

# 1. Load data
df = pd.read_excel(FILE_PATH)
print("=" * 65)
print("DISASTER TWEET CLASSIFICATION - NLP PROJECT")
print("=" * 65)
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nTarget distribution:\n", df["target"].value_counts())

# 2. Remove duplicates and handle missing values
df = df.drop_duplicates().copy()
for col in ["text", "keyword", "location"]:
    if col in df.columns:
        df[col] = df[col].fillna("" if col != "location" else "Unknown")

# 3. Class distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="target")
plt.title("Distribution of Disaster and Non-Disaster Tweets")
plt.xlabel("Target (0 = Non-Disaster, 1 = Disaster)")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150)
plt.close()

# 4. Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# 5. Additional features for EDA
df["tweet_length"] = df["text"].str.len()
df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
df["hashtag_count"] = df["text"].apply(lambda x: len(re.findall(r"#\w+", str(x))))
df["mention_count"] = df["text"].apply(lambda x: len(re.findall(r"@\w+", str(x))))
df["url_present"] = df["text"].apply(
    lambda x: int(bool(re.search(r"http\S+|www\S+", str(x))))
)

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="tweet_length", bins=40)
plt.title("Tweet Length Distribution")
plt.xlabel("Tweet Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/tweet_length_distribution.png", dpi=150)
plt.close()

# 6. Keyword analysis
keyword_counts = df[df["keyword"] != ""]["keyword"].value_counts().head(20)
keyword_counts.to_csv("outputs/top_keywords.csv", header=["frequency"])
print("\nTop keywords:\n", keyword_counts)

# 7. Train/test split
X = df["clean_text"]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 8. TF-IDF
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 9. Candidate models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=30, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=30, random_state=42, n_jobs=-1
    )
}

results = []
trained_models = {}

# 10. Train/evaluate
for name, model in models.items():
    print("\n" + "=" * 65)
    print("TRAINING:", name)
    print("=" * 65)

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    else:
        y_prob = model.decision_function(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))
    print("\nClassification Report:\n",
          classification_report(
              y_test, y_pred,
              target_names=["Non-Disaster", "Disaster"],
              zero_division=0
          ))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })
    trained_models[name] = model

results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON\n", results_df.round(4).to_string(index=False))
results_df.to_csv("outputs/model_results.csv", index=False)

# 11. Model comparison plot
ax = results_df.set_index("Model")[[
    "Accuracy", "Precision", "Recall", "F1 Score"
]].plot(kind="bar", figsize=(10, 6))
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.close()

# 12. Select model by highest F1 score
best_model_name = results_df.loc[results_df["F1 Score"].idxmax(), "Model"]
best_model = trained_models[best_model_name]
print("\nSelected model:", best_model_name)

best_predictions = best_model.predict(X_test_tfidf)

# 13. Confusion matrix
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Non-Disaster", "Disaster"],
    yticklabels=["Non-Disaster", "Disaster"]
)
plt.title("Confusion Matrix - " + best_model_name)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# 14. ROC curve
if hasattr(best_model, "predict_proba"):
    best_probability = best_model.predict_proba(X_test_tfidf)[:, 1]
else:
    best_probability = best_model.decision_function(X_test_tfidf)

fpr, tpr, _ = roc_curve(y_test, best_probability)
auc_score = roc_auc_score(y_test, best_probability)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - " + best_model_name)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150)
plt.close()

# 15. Precision-recall curve
p_values, r_values, _ = precision_recall_curve(y_test, best_probability)
plt.figure(figsize=(8, 6))
plt.plot(r_values, p_values)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - " + best_model_name)
plt.tight_layout()
plt.savefig("outputs/precision_recall_curve.png", dpi=150)
plt.close()

# 16. Feature importance
feature_names = np.array(tfidf.get_feature_names_out())

if hasattr(best_model, "coef_"):
    values = best_model.coef_[0]
    top_indices = np.argsort(np.abs(values))[-20:]
    top_features = feature_names[top_indices]
    top_values = values[top_indices]
    xlabel = "Model Coefficient"
else:
    values = best_model.feature_importances_
    top_indices = np.argsort(values)[-20:]
    top_features = feature_names[top_indices]
    top_values = values[top_indices]
    xlabel = "Importance"

plt.figure(figsize=(10, 7))
plt.barh(top_features, top_values)
plt.title("Top Important TF-IDF Features")
plt.xlabel(xlabel)
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()

pd.DataFrame({
    "feature": top_features,
    "importance_or_coefficient": top_values
}).sort_values(
    "importance_or_coefficient", key=np.abs, ascending=False
).to_csv("outputs/top_features.csv", index=False)

# 17. Save model and vectorizer
with open("disaster_tweet_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# 18. Sample predictions
sample_tweets = [
    "There is a massive earthquake in the city",
    "I love this song, it is a disaster",
    "Flood waters are rising and people are being evacuated",
    "My exam was a complete disaster"
]

sample_features = tfidf.transform([clean_text(x) for x in sample_tweets])
sample_predictions = best_model.predict(sample_features)

print("\n" + "=" * 65)
print("SAMPLE PREDICTIONS")
print("=" * 65)
for tweet, prediction in zip(sample_tweets, sample_predictions):
    print("\nTweet:", tweet)
    print("Prediction:", "REAL DISASTER" if prediction == 1 else "NON-DISASTER")

print("\nProject completed.")
print("Model saved as disaster_tweet_model.pkl")
print("Vectorizer saved as tfidf_vectorizer.pkl")
print("Charts/results saved inside outputs/")
