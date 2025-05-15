# api/chatbot.py
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data and model at startup
df = pd.read_excel("data.xlsx")[["Input", "Response"]].dropna()
model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = model.encode(df["Input"].tolist(), convert_to_tensor=False)

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: Query):
    query_embedding = model.encode([query.message])
    scores = cosine_similarity(query_embedding, question_embeddings)[0]
    best_idx = scores.argmax()
    response = df.iloc[best_idx]["Response"]
    return {"response": response}
