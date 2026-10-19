"""
Email Priority Scorer — Flask Application
Serves the web UI and handles email classification requests.
"""

import os
import pickle
import re
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load trained model and vectorizer
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# Priority labels and their metadata
PRIORITIES = {
    0: {"label": "Low Priority", "icon": "📋", "color": "#6b7280"},
    1: {"label": "Normal", "icon": "📬", "color": "#3b82f6"},
    2: {"label": "Important", "icon": "⚡", "color": "#f59e0b"},
    3: {"label": "Urgent", "icon": "🔴", "color": "#ef4444"},
}

# Keywords that signal priority (used to highlight reasoning)
PRIORITY_KEYWORDS = {
    3: ["urgent", "critical", "emergency", "immediately", "asap", "right now",
        "act now", "p0", "crash", "breach", "down", "broken", "fire", "drop everything"],
    2: ["deadline", "due", "action needed", "required", "approval", "review",
        "by friday", "by tomorrow", "end of week", "end of day", "confirm",
        "please respond", "blocking", "important"],
    1: ["fyi", "heads up", "update", "sharing", "check in", "notes",
        "quick question", "when you get a chance", "no rush"],
    0: ["newsletter", "unsubscribe", "fun", "picnic", "birthday", "merch",
        "optional", "just for fun", "meme", "playlist", "casual"],
}


def extract_signals(text):
    """Find keywords/phrases that influenced the prediction."""
    text_lower = text.lower()
    found = []
    for priority in [3, 2, 1, 0]:
        for keyword in PRIORITY_KEYWORDS[priority]:
            if keyword in text_lower:
                found.append({"word": keyword, "priority": priority})
    return found[:6]  # top 6 signals


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("text", "").strip()

    if not email_text:
        return jsonify({"error": "Please enter some email text."}), 400

    if len(email_text) < 10:
        return jsonify({"error": "Email text is too short. Enter at least a sentence."}), 400

    # Vectorize and predict
    X = vectorizer.transform([email_text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    confidence = float(np.max(probabilities)) * 100
    priority_info = PRIORITIES[int(prediction)]
    signals = extract_signals(email_text)

    # All priority scores for the breakdown
    breakdown = []
    for idx, prob in enumerate(probabilities):
        p = PRIORITIES[idx]
        breakdown.append({
            "label": p["label"],
            "icon": p["icon"],
            "color": p["color"],
            "score": round(float(prob) * 100, 1),
        })

    return jsonify({
        "priority": priority_info["label"],
        "icon": priority_info["icon"],
        "color": priority_info["color"],
        "confidence": round(confidence, 1),
        "signals": signals,
        "breakdown": breakdown,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
