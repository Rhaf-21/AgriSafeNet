import os
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pipeline = joblib.load("agriculture_safety_pipeline.joblib")

app = FastAPI()

class RequestData(BaseModel):
    text: str

@app.post("/predict")
def predict(data: RequestData):

    prediction = pipeline.predict([data.text])[0]

    confidence = float(
        max(pipeline.predict_proba([data.text])[0])
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Explain the classification result briefly."
                },
                {
                    "role": "user",
                    "content": f"""
Text: {data.text}

Predicted Label: {prediction}

Explain why this category was predicted.
"""
                }
            ]
        )

        explanation = response.choices[0].message.content

    except Exception as e:
        explanation = f"Explanation error: {e}"

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "explanation": explanation
    }
