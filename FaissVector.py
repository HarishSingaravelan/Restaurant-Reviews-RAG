from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

documents = []
for i, row in df.iterrows():
    doc = Document(
        page_content=row["Title"] + " " + row["Review"],
        metadata={"rating": row["Rating"], "date": row["Date"]}
    )
    documents.append(doc)

# Build FAISS index in memory (no sqlite, no persist issues)
vector_store = FAISS.from_documents(documents, embedding=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
