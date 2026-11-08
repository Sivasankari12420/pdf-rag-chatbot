
import faiss
import numpy as np
import json

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Step 1: Load PDF
loader = PyPDFLoader("data/UI+Testing-+Practice+Test.pdf")
pages = loader.load()


# Step 2: Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(pages)


# Step 3: Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Step 4: Generate embeddings for chunks
chunk_texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(chunk_texts)


# Step 5: Convert embeddings to NumPy float32
embeddings = np.array(embeddings).astype("float32")


# Step 6: Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)


# Step 7: Add embeddings to FAISS
index.add(embeddings)


# Step 8: Display information
print("Total chunks:", len(chunks))
print("Embedding dimensions:", dimension)
print("Vectors stored in FAISS:", index.ntotal)


# Step 9: Save FAISS index
faiss.write_index(index, "faiss_index.bin")

print("FAISS index saved successfully!")


# Step 10: Save chunks into JSON
chunk_data = [
    {
        "page_content": chunk.page_content,
        "metadata": chunk.metadata
    }
    for chunk in chunks
]

with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(
        chunk_data,
        file,
        ensure_ascii=False,
        indent=2
    )

print("Chunks saved successfully!")
