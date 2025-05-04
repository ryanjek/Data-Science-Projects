from typing import Tuple, List
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.documents import Document
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

# Structured Prompt for RAG QA
# Prevents hallucination by discouraging external knowledge
RAG_PROMPT_TEMPLATE = PromptTemplate.from_template("""
You are an intelligent assistant helping users answer questions based only on the provided document chunks.

Use **only** the text in the "Context" section below to answer the question.

Guidelines:
- If the answer is clearly present, provide a concise and accurate response.
- If the answer is partially present, summarize what can be inferred.
- If the answer is not present, reply exactly with: "I don't have enough information to answer this question."
- Do not make up any facts or use external knowledge.

---

Context:
{context}

---

Question: {question}

Answer:
""")

# RAG Chain
def perform_rag_qa(
    # User input
    question: str,
    # FAISS VectorDB
    vector_store,
    llm: LLM,
    # Default k set to 3
    k: int = 3
) -> Tuple[str, List[Document]]:
    """
    Perform Retrieval-Augmented Generation using LangChain PromptTemplate and structured context.
    """
    # Convert FAISS vector store as a retriever object
    # Find top k relevant document chunk, use cosine similiarity
    retriever: VectorStoreRetriever = vector_store.as_retriever(
        # Use cosine similarity for finding top k similar chunk
        search_type="similarity",
        search_kwargs={"k": k}
    )

    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs:
        return "I couldn't find relevant context to answer that question.", []

    # Build context.
    # Add header and page meta deta if available
    # Join into 1 single context for prompt
    # Explainability: User can see exactly which chunk used to generate the answer
    context_chunks = []
    for i, doc in enumerate(retrieved_docs):
        header = f"[Chunk {i+1}]"
        page = f" (Page {doc.metadata.get('page')})" if doc.metadata and 'page' in doc.metadata else ""
        chunk_text = f"{header}{page}:\n{doc.page_content.strip()}"
        context_chunks.append(chunk_text)

    context = "\n\n".join(context_chunks)

    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    # Send full prompt to LLM
    response = llm.invoke(prompt)
    answer = getattr(response, "content", str(response))

    return answer.strip(), retrieved_docs
