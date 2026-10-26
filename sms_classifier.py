"""Reusable SMS SVM pipeline. Adapted from Fuwad9096/svm_AI_project.

Keeping the transformer here lets joblib reload it in a separate process.
"""

import re
import string
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

NUMERIC_FEATURES = [
    "exclamation_count", "digit_count", "uppercase_ratio",
    "message_length", "has_url", "has_money_or_bait_word",
]
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
BAIT_PATTERN = re.compile(r"[£$€]|\b(?:free|win|won|prize|urgent)\b", re.IGNORECASE)


def setup_nltk():
    """Download only missing resources; fail clearly if the download fails."""
    for package, resources in {
        "stopwords": ("corpora/stopwords", "corpora/stopwords.zip"),
        "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
    }.items():
        found = False
        for resource in resources:
            try:
                nltk.data.find(resource)
                found = True
                break
            except LookupError:
                pass
        if not found and not nltk.download(package, quiet=True):
            raise RuntimeError(
                f"Could not download NLTK {package}. Connect to the internet and run "
                "python -m nltk.downloader stopwords wordnet"
            )
    # Check that both resources are usable, rather than trusting the downloader.
    stopwords.words("english")
    WordNetLemmatizer().lemmatize("messages")


def clean_text(text, stop_words=None, lemmatizer=None):
    """Clean text for TF-IDF; numeric features retain the removed raw signals."""
    if stop_words is None:
        stop_words = set(stopwords.words("english"))
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    text = URL_PATTERN.sub(" ", str(text).lower())
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    # Replace punctuation with spaces to avoid joining unrelated words.
    text = text.translate(str.maketrans({char: " " for char in string.punctuation}))
    return " ".join(
        lemmatizer.lemmatize(word) for word in text.split()
        if word not in stop_words
    )


def extract_spam_features(text):
    text = str(text)
    letters = [char for char in text if char.isalpha()]
    return {
        "exclamation_count": text.count("!"),
        "digit_count": sum(char.isdigit() for char in text),
        "uppercase_ratio": (
            sum(char.isupper() for char in letters) / len(letters) if letters else 0.0
        ),
        "message_length": len(text),
        "has_url": int(bool(URL_PATTERN.search(text))),
        "has_money_or_bait_word": int(bool(BAIT_PATTERN.search(text))),
    }


def _messages_from_input(X):
    if isinstance(X, str):
        raise TypeError("Pass a list of messages, for example model.predict([text]).")
    if isinstance(X, pd.DataFrame):
        if "message" not in X:
            raise ValueError("The DataFrame must contain a message column.")
        values = X["message"].tolist()
    else:
        values = list(X)
    if not all(isinstance(value, str) for value in values):
        raise ValueError("Every message must be a string; missing values are not accepted.")
    return values


class SMSFeatureBuilder(TransformerMixin, BaseEstimator):
    """Convert raw messages into cleaned text and six numeric features."""

    def fit(self, X, y=None):
        _messages_from_input(X)
        self.stop_words_ = set(stopwords.words("english"))
        self.lemmatizer_ = WordNetLemmatizer()
        return self

    def transform(self, X):
        if not hasattr(self, "stop_words_"):
            raise RuntimeError("Fit the feature builder before transforming messages.")
        rows = []
        for text in _messages_from_input(X):
            rows.append({
                "clean_message": clean_text(text, self.stop_words_, self.lemmatizer_),
                **extract_spam_features(text),
            })
        return pd.DataFrame(rows, columns=["clean_message", *NUMERIC_FEATURES])


def build_pipeline():
    features = ColumnTransformer([
        ("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), sublinear_tf=True,
        ), "clean_message"),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ], sparse_threshold=1.0)
    return Pipeline([
        ("raw_features", SMSFeatureBuilder()),
        ("features", features),
        ("svm", LinearSVC(class_weight="balanced", dual="auto",
                          max_iter=10000, random_state=42)),
    ])


def parameter_grid(quick=False):
    if quick:
        return {"svm__C": [0.1, 1.0]}
    return {
        "features__tfidf__ngram_range": [(1, 1), (1, 2)],
        "features__tfidf__max_features": [3000, 5000],
        "svm__C": [0.1, 1.0, 10.0],
        "svm__class_weight": [None, "balanced"],
    }


def load_dataset(path):
    """Read included CSV, Kaggle v1/v2 CSV, or UCI tab-separated collection."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
        if {"v1", "v2"}.issubset(df.columns):
            df = df.rename(columns={"v1": "label", "v2": "message"})
        if not {"label", "message"}.issubset(df.columns):
            raise ValueError("CSV needs label/message or v1/v2 columns.")
        df = df[["label", "message"]].copy()
    else:
        rows = [line.split("\t", 1) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if any(len(row) != 2 for row in rows):
            raise ValueError("UCI data needs one label<TAB>message per line.")
        df = pd.DataFrame(rows, columns=["label", "message"])

    raw_rows = len(df)
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    invalid = set(df["label"]) - {"ham", "spam"}
    if invalid:
        raise ValueError(f"Unknown labels: {sorted(invalid)}")
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() > 0].copy()
    if (df.groupby("message")["label"].nunique() > 1).any():
        raise ValueError("Identical messages have conflicting labels; resolve them first.")
    before_dedup = len(df)
    df = df.drop_duplicates(subset="message").reset_index(drop=True)
    df["target"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    if set(df["target"]) != {0, 1}:
        raise ValueError("Training requires both ham and spam examples.")
    df.attrs["dataset_summary"] = {
        "raw_rows": raw_rows,
        "invalid_or_empty_rows_removed": raw_rows - before_dedup,
        "duplicate_messages_removed": before_dedup - len(df),
        "usable_rows": len(df),
        "class_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
    }
    return df


def top_features(model, count=15):
    names = model.named_steps["features"].get_feature_names_out()
    weights = model.named_steps["svm"].coef_[0]
    order = np.argsort(weights)
    spam = pd.DataFrame({"feature": names[order[-count:][::-1]],
                         "weight": weights[order[-count:][::-1]]})
    ham = pd.DataFrame({"feature": names[order[:count]], "weight": weights[order[:count]]})
    return spam, ham


def predict_sms(model, text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty SMS message.")
    return "Spam" if int(model.predict([text])[0]) == 1 else "Ham"
