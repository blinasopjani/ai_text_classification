from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words("english"))

# Load model and vectorizer
model = joblib.load("log_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# FastAPI app
app = FastAPI()

# Input schema for multiple texts
class TextsInput(BaseModel):
    texts: list[str]

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# Predict category
def predict_category(text):
    text_clean = clean_text(text)
    X = vectorizer.transform([text_clean])
    return int(model.predict(X)[0])

@app.post("/predict")
def predict_endpoint(input: TextsInput):
    if not input.texts or not all(isinstance(t, str) and t.strip() for t in input.texts):
        raise HTTPException(status_code=400, detail="Texts cannot be empty or invalid.")
    categories = [predict_category(t) for t in input.texts]
    return {"categories": categories}
