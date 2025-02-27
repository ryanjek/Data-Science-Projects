import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

# Force model to use CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"  

# Cache model to avoid reloading on each rerun
@st.cache_resource
def initialize_huggingface_model(model_name="google/flan-t5-large"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")
    hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512, min_length = 10)
    return HuggingFacePipeline(pipeline=hf_pipeline)

# Load and split PDF efficiently
def load_and_split_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)  
    chunks = text_splitter.split_documents(documents)
    return [{"page_content": chunk.page_content[:512], "metadata": chunk.metadata} for chunk in chunks]

# Build vector store with FAISS
def build_vector_store(texts, model_name="sentence-transformers/all-mpnet-base-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    return vector_store

# Perform Retrieval-Augmented QA
def perform_rag_qa(question, vector_store, llm):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    result = qa_chain.invoke({"query": question})
    retrieved_text = " ".join([doc.page_content for doc in result["source_documents"]]) 
    prompt = f"Answer based only on the text:\n\n{retrieved_text}\n\nQ: {question}\nA:"
    return llm.invoke(prompt), result["source_documents"]


# Streamlit UI
st.title("📄 PDF Q&A Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    st.info("Processing your PDF...")
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    pdf_chunks = load_and_split_pdf("temp.pdf")
    st.success(f"✅ PDF successfully loaded and split into {len(pdf_chunks)} chunks!")

    # Build vector store
    vector_store = build_vector_store([chunk["page_content"] for chunk in pdf_chunks])

    # Initialize LLM
    llm = initialize_huggingface_model()

    question = st.text_input("🔍 Ask a question about the PDF:")
    if question:
        st.info("Retrieving relevant chunks and generating an answer...")
        answer, sources = perform_rag_qa(question, vector_store, llm)

        st.subheader("📌 Answer:")
        st.write(answer)

        st.subheader("📜 Relevant Chunks:")
        for source in sources:
            metadata = source.metadata if "metadata" in source else "N/A"
            st.write(f"📄 Page Metadata: {metadata}")
            st.write(source.page_content)


# streamlit run /Users/ryan/Documents/GitHub/Data-Science-Projects/Chatbot/Summariser_test1.py
