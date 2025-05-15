import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
def load_data():
    return pd.read_excel("data.xlsx")[["Input", "Response"]].dropna()

df = load_data()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all questions once
question_embeddings = model.encode(df["Input"].tolist(), convert_to_tensor=True)

# Search function
def get_best_response(query):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, question_embeddings)[0]
    best_idx = scores.argmax()
    return df.iloc[best_idx]["Response"]

# Simple chat loop
print("Assalamu Alaikum 🌿! I'm your Bulipe Tech chatbot. Ask me anything about our digital skills programs. (Type 'exit' to quit)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye! 👋")
        break
    answer = get_best_response(user_input)
    print("Chatbot:", answer)
