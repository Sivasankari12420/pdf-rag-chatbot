
import faiss
import numpy as np
import json
import ollama

from sentence_transformers import SentenceTransformer


# Step 1: Load saved chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)


# Step 2: Load FAISS index
index = faiss.read_index("faiss_index.bin")


# Step 3: Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Step 4: Create RAG function
def ask_question(question):

    # Convert question into embedding
    question_embedding = model.encode([question])

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    # Search relevant chunks
    distances, indices = index.search(
        question_embedding,
        k=2
    )

    # Retrieve PDF context
    retrieved_chunks = []

    for chunk_index in indices[0]:
        retrieved_chunks.append(
            chunks[chunk_index]["page_content"]
        )

    # Combine chunks
    context = "\n\n".join(retrieved_chunks)

    # Create prompt
    prompt = f"""
You are a PDF question-answering assistant.

Answer using the information in the context.
If the answer is unavailable, say so.
Keep your answer short and clear.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate answer using Ollama
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Return answer
    return response["message"]["content"]
