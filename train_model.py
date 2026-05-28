from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_DATASET = BASE_DIR / "data" / "spam.csv"


def load_sms_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load common SMS spam CSV formats and normalize them to label/text columns."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Place spam.csv in data/ or pass a path."
        )

    try:
        dataframe = pd.read_csv(dataset_path, encoding="latin-1")
    except UnicodeDecodeError:
        dataframe = pd.read_csv(dataset_path)

    if {"v1", "v2"}.issubset(dataframe.columns):
        dataframe = dataframe.rename(columns={"v1": "label", "v2": "text"})
    elif not {"label", "text"}.issubset(dataframe.columns):
        raise ValueError("CSV must contain either v1/v2 columns or label/text columns.")

    dataframe = dataframe[["label", "text"]].dropna()
    dataframe["label"] = dataframe["label"].str.strip().str.lower()
    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe = dataframe[dataframe["label"].isin(["ham", "spam"])]

    if dataframe.empty:
        raise ValueError("No valid ham/spam rows found in the dataset.")

    return dataframe


def train_and_save_model(dataset_path: Path = DEFAULT_DATASET) -> None:
    dataframe = load_sms_dataset(dataset_path)

    x_train, x_test, y_train, y_test = train_test_split(
        dataframe["text"],
        dataframe["label"],
        test_size=0.2,
        random_state=42,
        stratify=dataframe["label"],
    )

    # TF-IDF converts text to weighted numeric features for Naive Bayes.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
    )
    classifier = MultinomialNB(alpha=0.1)

    pipeline = Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    print(classification_report(y_test, predictions, digits=4))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["classifier"], MODEL_DIR / "spam_model.pkl")
    joblib.dump(pipeline.named_steps["vectorizer"], MODEL_DIR / "vectorizer.pkl")
    print(f"Saved model files to {MODEL_DIR}")


if __name__ == "__main__":
    train_and_save_model()
