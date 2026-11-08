# PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF documents using semantic search and a locally running Large Language Model (LLM).

## Project Overview

This project allows users to upload or prepare PDF documents, retrieve relevant information using semantic search, and generate contextual answers using Ollama.

The application combines document processing, embeddings, vector search, and LLM-based response generation in a Streamlit interface.

## Technologies Used

- **Python** – Application development
- **LangChain Community** – PDF document loading
- **LangChain Text Splitters** – Text chunking
- **Sentence Transformers** – Text embeddings
- **FAISS** – Vector similarity search
- **Ollama** – Local LLM inference
- **Llama 3.2 (1B)** – Language model
- **Streamlit** – Web application interface
- **NumPy** – Numerical operations

## Architecture

```text
              PDF Document
                   |
                   v
             PDF Loader
                   |
                   v
              Text Chunking
                   |
                   v
          Sentence Transformer
                   |
                   v
           Generate Embeddings
                   |
                   v
             FAISS Index
                   |
                   v
            Semantic Search
                   |
             User Question
                   |
                   v
          Retrieve Relevant Chunks
                   |
                   v
          Context + User Question
                   |
                   v
              Ollama LLM
                   |
                   v
           Generated Answer
                   |
                   v
           Streamlit UI
```

## Features

- PDF document processing
- Text chunking with overlap
- Semantic embeddings using Sentence Transformers
- FAISS vector similarity search
- Question answering using retrieved PDF context
- Local LLM inference using Ollama
- Interactive Streamlit chatbot
- Conversation history
- Clear chat functionality
- Retrieved context visibility with PDF page references

## Project Structure

```text
pdf-rag-chatbot/
│
├── data/
│   └── UI+Testing-+Practice+Test.pdf
│
├── app.py
├── chatbot.py
├── llm_test.py
├── load_pdf.py
├── split_text.py
├── vector_store.py
├── search_faiss.py
├── rag_pipeline.py
│
├── chunks.json
├── faiss_index.bin
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sivasankari12420/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Ollama

Download Ollama:

https://ollama.com/download

Pull the LLM:

```bash
ollama pull llama3.2:1b
```

### 6. Prepare the PDF

Place your PDF document inside the `data` directory.

If the sample PDF is not included in the repository, add your own PDF document and update the PDF path in `vector_store.py`.

### 7. Build the vector database

```bash
python vector_store.py
```

This generates:

- `chunks.json`
- `faiss_index.bin`

### 8. Run the application

```bash
streamlit run app.py
```

Open the application:

http://localhost:8501

## How RAG Works

1. Load the PDF document.
2. Extract text from PDF pages.
3. Split the text into smaller chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in the FAISS vector index.
6. Convert the user's question into an embedding.
7. Retrieve the most relevant chunks.
8. Send retrieved context and the question to Ollama.
9. Generate a contextual answer.
10. Display the answer and retrieved sources in Streamlit.

## Example Questions

- Who should review the UI design?
- Against what should the design be reviewed?
- What recommendations are provided for checkout page testing?

The chatbot retrieves relevant information from the PDF and generates an answer using the local LLM.

## Learning Outcomes

- Understanding Retrieval-Augmented Generation (RAG)
- Working with PDF document loaders
- Understanding text chunking and overlap
- Generating embeddings using Sentence Transformers
- Performing semantic search using FAISS
- Integrating a local LLM using Ollama
- Building an interactive application using Streamlit
- Understanding the complete document question-answering pipeline

## Future Enhancements

- Support multiple PDF documents
- Allow PDF uploads directly through Streamlit
- Improve retrieval accuracy
- Add conversation memory
- Implement source citations and better page references
- Explore advanced RAG techniques
- Integrate evaluation metrics for RAG responses

## Author

**Sivasankari**

GitHub: https://github.com/Sivasankari12420
