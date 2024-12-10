from app import app
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    # Load the .env file
    load_dotenv()

    # Access the API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    app()