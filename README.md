# 📧 Email Priority Scorer

An ML-powered web application that classifies emails by priority level using Natural Language Processing.

## What it does

Paste any email text and the model predicts its priority:
- 🔴 **Urgent** — needs immediate attention
- ⚡ **Important** — action needed soon
- 📬 **Normal** — standard communication
- 📋 **Low Priority** — newsletters, FYI, social

It also shows a confidence score, a breakdown across all categories, and highlights the key words/phrases that influenced the decision.

## Tech Stack

- **Python 3.11** — core language
- **Flask** — lightweight web framework
- **scikit-learn** — ML model (Logistic Regression + TF-IDF)
- **Docker** — containerization

## How to Run with Docker

### 1. Build the Docker image
```bash
docker build -t email-priority-scorer .
```

### 2. Run the container
```bash
docker run -p 5000:5000 email-priority-scorer
```

### 3. Open in browser
```
http://localhost:5000
```

That's it! The model trains automatically during the Docker build step.

## How the ML Model Works

1. **Training Data** — A curated synthetic dataset of 100 labeled emails across 4 priority levels
2. **Feature Extraction** — TF-IDF vectorizer converts email text into numerical features using unigrams and bigrams
3. **Model** — Logistic Regression classifier with balanced class weights
4. **Prediction** — Input email is vectorized and classified, returning the priority label with probability scores

## Project Structure

```
email-priority-scorer/
├── app.py                  # Flask server + prediction endpoint
├── model/
│   └── train_model.py      # Model training script
├── templates/
│   └── index.html          # Web UI
├── static/
│   └── style.css           # Styling
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
├── .dockerignore           # Files excluded from build
└── README.md               # This file
```
