import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_PWD = os.getenv('NEO4J_PWD')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')