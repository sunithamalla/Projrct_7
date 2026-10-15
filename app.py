from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

with open("disaster_tweet_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    tweet = ""

    if request.method == "POST":
        tweet = request.form.get("tweet", "").strip()

        if tweet:
            features = vectorizer.transform([clean_text(tweet)])
            result = int(model.predict(features)[0])

            if hasattr(model, "predict_proba"):
                probability = round(float(model.predict_proba(features)[0][1]) * 100, 2)

            prediction = (
                "REAL DISASTER TWEET"
                if result == 1
                else "NON-DISASTER TWEET"
            )

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        tweet=tweet
    )

if __name__ == "__main__":
    app.run(debug=True)
