# PDF Question Answering Assistant (RAG Implementation)

This project is a Retrieval-Augmented Generation (RAG) application that allows users to ask natural language questions about the content of an uploaded PDF document. It uses semantic similarity search and modern language models to return accurate, context-grounded answers.

The application is built using:

- Streamlit for the user interface
- LangChain for orchestration and abstraction
- HuggingFace and OpenAI models for language generation
- FAISS for efficient vector similarity search
- Sentence Transformers for semantic embeddings

---

## Features

- Upload and analyse any PDF document
- Automatically extracts and segments document content
- Embeds text chunks using a sentence transformer model
- Stores embeddings in a FAISS vector store for fast retrieval
- Retrieves top-k most relevant chunks for a user query
- Generates responses using a selected LLM (local or remote)
- Displays source content used in generating the answer

---

## System Overview

### Workflow

1. **PDF Ingestion**
   - Extract and clean text from the uploaded PDF file.

2. **Chunking**
   - The document is split into overlapping text chunks (default: 1000 characters with 200-character overlap) to respect the language model's context window.

3. **Embedding**
   - Each chunk is embedded into a fixed-size vector using the `all-mpnet-base-v2` model.
   - Vectors represent the semantic meaning of the chunk.

4. **Vector Store (FAISS)**
   - Embeddings are stored in a FAISS index.
   - Enables fast nearest-neighbour retrieval during question answering.

5. **Query Handling**
   - The user submits a question.
   - The question is embedded and used to search for top-k similar chunks in the FAISS index.

6. **Prompt Assembly**
   - Retrieved chunks are assembled into a structured prompt along with the user’s question.

7. **Answer Generation**
   - The prompt is sent to an LLM (either HuggingFace or OpenAI) to generate an answer grounded in the retrieved context.
