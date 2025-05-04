import os
import logging
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI



load_dotenv()
logging.basicConfig(level=logging.INFO)

# Check if GPU (cuda), Apple (mps) or cpu for running the model
def get_device() -> str:
    """Determine the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Google Flan T5 Large model
def initialize_llm_model(model_name: str = "google/flan-t5-large") -> HuggingFacePipeline:
    """
    Load a HuggingFace transformer model into a LangChain-compatible pipeline.
    """
    device = get_device()
    logging.info(f"Loading model '{model_name}' on device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

        hf_pipeline = pipeline(
            # Used for tasks like QA, summarisation
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            # Max num of token in model output
            max_length=512,
            # No random sampling for factual QA
            do_sample=False)

        return HuggingFacePipeline(pipeline=hf_pipeline)

    except Exception as e:
        logging.error(f"Failed to load HuggingFace model: {e}")
        raise RuntimeError(f"Could not load HuggingFace model '{model_name}'. Check your model name or system memory.")

# Open AI model
def initialize_llm_model_openai(model_name: str = "gpt-3.5-turbo", temperature: float = 0.0) -> ChatOpenAI:
    """
    Load OpenAI Chat model into a LangChain-compatible object.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your environment or .env file.")
    
    # Temperature set to 0 for factual QA
    logging.info(f"Loading OpenAI model '{model_name}' with temperature {temperature}")
    return ChatOpenAI(model_name=model_name, temperature=temperature)


