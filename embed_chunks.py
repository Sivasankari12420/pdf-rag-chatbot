
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load the PDF
loader = PyPDFLoader("data/UI+Testing-+Practice+Test.pdf")
pages = loader.load()

# Step 2: Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(pages)

# Step 3: Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Extract text from chunks
chunk_texts = [chunk.page_content for chunk in chunks]

# Step 5: Generate embeddings
embeddings = model.encode(chunk_texts)

# Step 6: Display results
print("Total chunks:", len(chunks))
print("Embeddings shape:", embeddings.shape)

# Display first chunk embedding
print("\nFirst chunk embedding:")
print(embeddings[0])



import matplotlib.pyplot as plt

# Plot embeddings of all 4 chunks
plt.figure(figsize=(12, 5))

for i in range(len(embeddings)):
    plt.plot(embeddings[i], label=f"Chunk {i + 1}")

plt.title("PDF Chunk Embeddings")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Embedding Values")
plt.legend()
plt.grid()

plt.show()
