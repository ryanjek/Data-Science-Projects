from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
import torch


file_path = "/Users/ryan/Documents/GitHub/Data-Science-Projects/PDF/LLM.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Set up Falcon-7B
model = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model)
model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float32, device_map="cpu")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)

# Retriever and QA chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=model_pipeline, retriever=retriever, return_source_documents=True)

# Ask a question
query = "What is the document about?"
response = qa_chain.run(query)
print("Answer:", response["answer"])

