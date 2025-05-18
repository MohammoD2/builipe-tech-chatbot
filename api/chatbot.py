import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Chatbot:
    def __init__(self, data_path="data.xlsx", model_name="all-MiniLM-L6-v2"):
        self.df = pd.read_excel(data_path)[["Input", "Response"]].dropna()
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(self.df["Input"].tolist(), convert_to_tensor=True)

    def get_response(self, query: str) -> str:
        query_embedding = self.model.encode([query])
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        best_idx = scores.argmax()
        return self.df.iloc[best_idx]["Response"]