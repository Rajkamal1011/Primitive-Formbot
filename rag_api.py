from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data and embedding model
input_file = "chunks_and_schemes_with_embeddings.xlsx"
data = pd.read_excel(input_file)

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")

# Request model
class QueryRequest(BaseModel):
    form_entry: str  # Currently not used but included as per original code
    voice_query: str
    scheme_name: str

# Function to compute dot product similarity
def dot_product_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2)

# RAG pipeline function
def rag_pipeline(form_entry, voice_query, scheme_name):
    # Filter chunks related to the given scheme_name
    scheme_name = scheme_name.strip()
    scheme_data = data[data['Name'].str.lower() == scheme_name.lower()]

    # Encode the query
    query_embedding = embedding_model.encode(voice_query)

    # Compute similarity scores
    scheme_data['Similarity'] = scheme_data['Chunk Embedding'].apply(
        lambda emb: dot_product_similarity(query_embedding, np.array(eval(emb)))
    )

    # Select top 3 similar chunks
    top_chunks = scheme_data.sort_values(by='Similarity', ascending=False).head(3)['Chunks'].tolist()

    # Prepare the prompt
    prompt = (
        f"With respect to the following information:\n"
        f"{chr(10).join(top_chunks)}\n"
        f"Please answer the following query:\n{voice_query}"
    )

    # OpenAI API call using ChatCompletion
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Adjust based on your model preference
        messages=[
            {"role": "system", "content": "You are a helpful assistant for form filling and schemes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400
    )

    return response.choices[0].message['content'].strip()

# API route
@app.post("/get_llm_response_schemes")
async def get_rag_response(request: QueryRequest):
    response = rag_pipeline(request.form_entry, request.voice_query, request.scheme_name)
    return {"query":request.voice_query,"response": response,"status": 200}
