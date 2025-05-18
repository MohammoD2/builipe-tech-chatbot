from fastapi import FastAPI, Request
from pydantic import BaseModel, constr
from chatbot import Chatbot
import logging

app = FastAPI()
bot = Chatbot()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Query(BaseModel):
    message: constr(min_length=1, max_length=500)

@app.get("/")
def root():
    return {"message": "Welcome to Bulipe Tech Chatbot API!"}

@app.post("/chat/")
def chat(query: Query):
    logger.info(f"Received query: {query.message}")
    response = bot.get_response(query.message)
    return {"response": response}