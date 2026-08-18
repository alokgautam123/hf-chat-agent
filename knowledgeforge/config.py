import os


# Directories
DOCS_DIR = "docs"
VECTOR_DB_DIR = "chroma_db"
MODELS_DIR = "models"
AUTH_DB_PATH = "data/knowledgeforge.db"

# Authentication
SESSION_SECRET = os.environ.get("SESSION_SECRET")
SESSION_COOKIE_SECURE = (
    os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
)
SESSION_MAX_AGE = int(os.environ.get("SESSION_MAX_AGE", "604800"))

# ChromaDB
COLLECTION_NAME = "knowledge_base"

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
TOP_K_RESULTS = 5

SIMILARITY_THRESHOLD = 1.20
MAX_DISTANCE = 1.2
