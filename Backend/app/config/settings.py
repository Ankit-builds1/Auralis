import os

from dotenv import load_dotenv


load_dotenv()


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

ML_SERVICE_URL = os.getenv(
    "ML_SERVICE_URL",
    "http://localhost:9000"
)