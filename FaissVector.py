from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import pandas as pd

# Load your data
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Use updated embedding interface
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Prepare documents
documents = [
    Document(
        page_content=row["Title"] + " " + row["Review"],
        metadata={"rating": row["Rating"], "date": row["Date"]}
    )
    for _, row in df.iterrows()
]

# Create FAISS index
vector_store = FAISS.from_documents(documents, embedding=embeddings)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
