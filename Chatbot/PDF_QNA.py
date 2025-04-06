from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import create_retrieval_chain
import torch
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

file_path = "/Users/ryan/Documents/GitHub/Data-Science-Projects/PDF/LLM.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Set up LLM
model = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model)
model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")
llm = HuggingFacePipeline(pipeline=model_pipeline) 

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)

# Retriever and QA chain
retriever = vectorstore.as_retriever()

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

query = "What is this document about?" 
context = "You are to summarise the document."

response = chain.invoke({"input": query})

print(response)

