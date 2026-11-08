
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF
loader = PyPDFLoader("data/UI+Testing-+Practice+Test.pdf")

# Extract the pages
pages = loader.load()

# Display the number of pages
print("Total pages:", len(pages))

# Display the content of the first page
print(pages[0].page_content)
