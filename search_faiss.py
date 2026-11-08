
import faiss
import numpy as np
import json

from sentence_transformers import SentenceTransformer


# Step 1: Load saved chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)


# Step 2: Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Step 3: Load saved FAISS index
index = faiss.read_index("faiss_index.bin")


# Step 4: Enter question
question = "Who should review the UI design?"


# Step 5: Convert question into embedding
question_embedding = model.encode([question])


# Step 6: Convert embedding into float32
question_embedding = np.array(
    question_embedding
).astype("float32")


# Step 7: Search FAISS
distances, indices = index.search(
    question_embedding,
    k=2
)


# Step 8: Display search results
print("\nQuestion:", question)

print("\nRelevant PDF chunks:")

for i in range(len(indices[0])):

    chunk_index = indices[0][i]

    distance = distances[0][i]

    print("\nChunk index:", chunk_index)

    print("Distance:", distance)

    print("Text:")

    print(chunks[chunk_index]["page_content"])
