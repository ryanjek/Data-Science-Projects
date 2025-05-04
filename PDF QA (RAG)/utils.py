from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional
from typing import List
import logging
import re
import os

logging.basicConfig(level=logging.INFO)

# Normalise noisy PDF
def clean_text(text: str) -> str:
    # Replace special characters
    text = text.replace('\xa0', ' ')
    # Collapse mutliple whitespaces into 1 space
    text = re.sub(r'\s+', ' ', text)
    # Trim leading/trailing spaces
    return text.strip()

def load_and_split_pdfs(pdf_paths: List[str]) -> List:
    """
    Load multiple PDFs and split them into token-approximate chunks for LLM use.
    """
    all_chunks = []
    try:
        for pdf_path in pdf_paths:
            # Use PyPDFLoader to load PDF file
            loader = PyPDFLoader(pdf_path)
            # Each page is a Document object
            documents = loader.load()

            # Clean each page content
            for doc in documents:
                doc.page_content = clean_text(doc.page_content)
                # Add source metadata to trace back to original file
                doc.metadata['source'] = os.path.basename(pdf_path)

            # Break long text into overlapping chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,          
                chunk_overlap=200,
                # Prefer to split on paragraph → line → word    
                separators=["\n\n", "\n", " "],
                # Use character length instead of token count to approximate size
                length_function=len       
            )

            # Split each document into overlapping chunks
            chunks = text_splitter.split_documents(documents)
            logging.info(f"{os.path.basename(pdf_path)} split into {len(chunks)} chunks.")
            all_chunks.extend(chunks)

        logging.info(f"Total combined chunks: {len(all_chunks)}")
        return all_chunks

    except Exception as e:
        logging.error(f"Error while loading/splitting PDFs: {e}")
        return []

    
def build_vector_store(chunks: List, embedding_model: str = "sentence-transformers/all-mpnet-base-v2") -> Optional[FAISS]:
    """
    Build a FAISS vector store from the provided chunks.

    Args:
        chunks: List of document chunks.
        embedding_model: Sentence-transformers model to use for embeddings.

    Returns:
        FAISS vector store or None on failure.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vector_store = FAISS.from_documents(chunks, embeddings)
        logging.info("Vector store created successfully.")
        return vector_store

    except Exception as e:
        logging.error(f"Error while building vector store: {e}")
        return None