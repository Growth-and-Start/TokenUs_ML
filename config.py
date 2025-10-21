import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv(".env")

class Config:

    DEBUG = True

    # AWS
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

    # MySQL
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3307))
    MYSQL_USER = os.getenv("FLASK_DB_USER", "root")
    MYSQL_PASSWORD = os.getenv("FLASK_DB_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("FLASK_DB_NAME", "")

    # Spring Boot Server URL
    BACKEND_URL = os.getenv("BACKEND_URL", "")
    API_PATH = os.getenv("API_PATH", "")

    # FAISS Storage Path
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index/faiss.index")

    # Download Folder
    DOWNLOAD_FOLDER = os.getenv("DOWNLOAD_FOLDER", "downloads")

    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))