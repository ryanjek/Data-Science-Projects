from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional
from typing import List
import logging
import re

logging.basicConfig(level=logging.INFO)

# Normalise noisy PDF
def clean_text(text: str) -> str:
    # Replace special characters
    text = text.replace('\xa0', ' ')
    # Collapse mutliple whitespaces into 1 space
    text = re.sub(r'\s+', ' ', text)
    # Trim leading/trailing spaces
    return text.strip()

def load_and_split_pdf(pdf_path: str) -> List:
    """
    Load a PDF and split it into token-approximate chunks for LLM use.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        # Each page is a Document object
        documents = loader.load()

        # Clean each page content
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)

        # Use a character-based splitter with approximate token logic
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,          
            chunk_overlap=200,         
            separators=["\n\n", "\n", " "],
            length_function=len       
        )

        chunks = text_splitter.split_documents(documents)
        logging.info(f"PDF split into {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        logging.error(f"Error while loading/splitting PDF: {e}")
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