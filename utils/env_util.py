import os
from dotenv import load_dotenv
load_dotenv()

class EnvironmentVariables:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")