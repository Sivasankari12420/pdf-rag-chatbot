
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Sample text
text = "Python is a programming language."

# Generate embedding
embedding = model.encode(text)

# Display the embedding
print(embedding)

# Display embedding dimensions
print("Embedding dimensions:", embedding.shape)
