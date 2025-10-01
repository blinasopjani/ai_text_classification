# AI Text Classification

This project is a **machine learning text classification API** built with **FastAPI**.  
It uses a trained **Logistic Regression model** with **TF-IDF features** to classify input text into categories.  

This project includes a **Jupyter Notebook** that demonstrates the full AI text classification workflow:


### Notebook Features

1. **Import Libraries** – pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, joblib.  
2. **Data Loading** – Reads training and test CSV files.  
3. **Data Preparation** – Combines title and description into a single column, renames labels.  
4. **Exploratory Analysis** – Examines sample counts per category and visualizes distributions.  
5. **Text Cleaning** – Lowercasing, removing non-alphabetic characters, and removing stopwords.  
6. **Vectorization** – Converts text to numerical features using TF-IDF.  
7. **Model Training** – Logistic Regression & Naive Bayes classifiers.  
8. **Evaluation** – Accuracy, precision, recall, F1-score, and confusion matrices.  
9. **Model Saving** – Uses Joblib to save vectorizer and models for reuse.  
10. **Prediction Functions** – Predict category for single or multiple sentences.  

The notebook demonstrates **end-to-end workflow** from raw data to a working API-ready model.

---

##  Features (*machine learning text classification API*)
- REST API using **FastAPI**  
- Model persistence with **Joblib**  
- Simple JSON input/output format  
- Easy to run locally  

---

##  Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/blinasopjani/ai_text_classification.git
cd ai_text_classification
```
### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the API
```bash
uvicorn app:app --reload
```
## By default, the API will be available at:
```bashhttp://127.0.0.1:8000```

##  API Usage

**Endpoint:**  /predict <br>
**Method:**  POST <br>
**Headers:** <br>
Content-Type: application/json

**Example Request:**  
```bash
```json
{
  "texts": [
    "The stock market crashed today.",
    "The new movie was a huge success.",
    "Local elections are coming soon."
  ]
}
```

{
  "categories": [3, 4, 1]
}

##  Model Explanation

- **TF-IDF Vectorizer:** Converts raw text into numerical feature vectors representing word importance.  
- **Logistic Regression:** A linear classifier trained to predict the category of input text based on TF-IDF features.  
- The model and vectorizer are saved with **Joblib** and loaded during API runtime.  

---

##  (Optional) Deploy on Render

This project can also be deployed on [Render](https://render.com).  

**Build Command:**  
```bash
pip install -r requirements.txt
```
**Start Command:**
```bash 
uvicorn app:app --host 0.0.0.0 --port $PORT
```









