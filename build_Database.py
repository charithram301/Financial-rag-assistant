import os
import fitz
import chromadb

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# Configuration
# -----------------------------

PDF_FOLDER = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "financial_reports"

# -----------------------------
# Load Embedding Model
# -----------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# -----------------------------
# Read PDFs
# -----------------------------

documents = []

for filename in os.listdir(PDF_FOLDER):

    if filename.endswith(".pdf"):

        filepath = os.path.join(PDF_FOLDER, filename)

        print(f"Reading {filename}")

        pdf = fitz.open(filepath)

        text = ""

        for page in pdf:
            text += page.get_text()

        documents.append({
            "company": filename.replace(".pdf", ""),
            "text": text
        })

print(f"Loaded {len(documents)} PDF files")

# -----------------------------
# Split into chunks
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []

for doc in documents:

    split_text = splitter.split_text(doc["text"])

    for chunk in split_text:

        chunks.append({
            "company": doc["company"],
            "text": chunk
        })

print("Total Chunks:", len(chunks))

# -----------------------------
# Create Embeddings
# -----------------------------

texts = [chunk["text"] for chunk in chunks]

print("Generating embeddings...")

embeddings = embedding_model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# -----------------------------
# Create Persistent ChromaDB
# -----------------------------

print("Creating ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

# Delete old collection if it exists
try:
    client.delete_collection(COLLECTION_NAME)
    print("Old collection deleted.")
except:
    pass

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# -----------------------------
# Store Documents
# -----------------------------

ids = [str(i) for i in range(len(chunks))]

metadatas = [
    {"company": chunk["company"]}
    for chunk in chunks
]

collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("=" * 50)
print("Database Created Successfully!")
print(f"Collection : {COLLECTION_NAME}")
print(f"Documents  : {len(texts)}")
print(f"Location   : {CHROMA_PATH}")
print("=" * 50)
