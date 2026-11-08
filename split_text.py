
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the PDF

loader = PyPDFLoader("data/UI+Testing-+Practice+Test.pdf")
pages = loader.load()

# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split the documents
chunks = text_splitter.split_documents(pages)

# Display the total number of chunks
print("Total chunks:", len(chunks))

# Display the first chunk
print(chunks[0].page_content)

# Display metadata of the first chunk
print(chunks[0].metadata)
