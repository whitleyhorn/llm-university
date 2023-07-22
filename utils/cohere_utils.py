from dotenv import load_dotenv
import cohere
import os

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
