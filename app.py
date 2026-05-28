import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
from flask import Flask, g, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATABASE_PATH = BASE_DIR / "spam_checker.db"

app = Flask(__name__)


def load_artifact(filename: str):
    artifact_path = MODEL_DIR / filename
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing {artifact_path}. Run `python train_model.py` before starting Flask."
        )
    return joblib.load(artifact_path)


model = load_artifact("spam_model.pkl")
vectorizer = load_artifact("vectorizer.pkl")


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS message_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_text TEXT NOT NULL,
                classification_label TEXT NOT NULL,
                spam_probability REAL NOT NULL,
                logged_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def fetch_recent_logs(limit: int = 25) -> list[sqlite3.Row]:
    cursor = get_db().execute(
        """
        SELECT id, raw_text, classification_label, spam_probability, logged_at
        FROM message_logs
        ORDER BY logged_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return cursor.fetchall()


def log_prediction(raw_text: str, label: str, spam_probability: float) -> None:
    get_db().execute(
        """
        INSERT INTO message_logs (raw_text, classification_label, spam_probability, logged_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            raw_text,
            label,
            spam_probability,
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    get_db().commit()


@app.route("/", methods=["GET"])
def index():
    logs = fetch_recent_logs()
    return render_template("index.html", logs=logs)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Message text is required."}), 400

    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    class_to_probability = dict(zip(model.classes_, probabilities))
    spam_probability = float(class_to_probability.get("spam", 0.0))

    is_spam = prediction == "spam"
    confidence = spam_probability if is_spam else 1.0 - spam_probability
    label = "Spam" if is_spam else "Ham"

    log_prediction(text, label, spam_probability)

    return jsonify(
        {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "is_spam": is_spam,
            "spam_probability": round(spam_probability * 100, 2),
        }
    )


init_db()


if __name__ == "__main__":
    app.run(debug=True)
