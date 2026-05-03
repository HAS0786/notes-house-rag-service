import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = r"D:\Notes House FYP\noteshouse_backend\data\temp"
FAISS_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")
DOCS_PATH = os.path.join(BASE_DIR, "data", "docs.pkl")

OPENAI_API_KEY = "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"

CHUNK_SIZE = 280
TOP_K = 70

TESSERACT_PATH = r"D:\Notes House FYP\tools\tesseract\tesseract.exe"

os.makedirs(UPLOAD_DIR, exist_ok=True)