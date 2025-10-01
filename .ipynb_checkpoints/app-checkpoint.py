# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# Load model and vectorizer
log_model = joblib.load("log_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize FastAPI
app = FastAPI(title="AI Text Classification")

# Pydantic model for request
class TextsRequest(BaseModel):
    texts: list[str]

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# Predict function
def predict_category(text):
    text_clean = clean_text(text)
    X = vectorizer.transform([text_clean])
    return int(log_model.predict(X)[0])

# Root route (optional)
@app.get("/")
def root():
    return {"message": "AI Text Classification API is running. Use /predict endpoint."}

# Predict route
@app.post("/predict")
def predict(request: TextsRequest):
    predictions = [predict_category(t) for t in request.texts]
    return {"categories": predictions}
