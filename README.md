# Agriculture Safety Classifier

## Project Overview
An NLP-based system that classifies agricultural safety-related queries using TF-IDF and Logistic Regression, with deployment via FastAPI and an interactive Streamlit interface.

## Project Workflow
User Input → TF-IDF Vectorization → Logistic Regression Model → Prediction → OpenAI Explanation → Streamlit UI

## Components
- Built an NLP classification system using TF-IDF and Logistic Regression for agricultural safety queries.
- Improved performance through baseline testing and hyperparameter tuning.
- Deployed the model via a FastAPI service for real-time predictions.
- Integrated OpenAI to generate explanations and agricultural guidance.
- Developed a Streamlit interface for user interaction and results display.

## Tech Stack
Python, Scikit-learn, Pandas, NumPy, FastAPI, Streamlit, OpenAI API

## How to Run
```bash
pip install -r requirements.txt
```

```bash
uvicorn main:app --reload
```

```bash
streamlit run app.py
```

## API Endpoint
- /predict → Returns classification result
