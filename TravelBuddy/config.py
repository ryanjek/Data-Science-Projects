from dotenv import load_dotenv
import os

load_dotenv()

# --- SET API KEYS ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not OPENAI_API_KEY or not GOOGLE_MAPS_API_KEY:
    raise ValueError("API keys are missing! Ensure they are set in the .env file.")
