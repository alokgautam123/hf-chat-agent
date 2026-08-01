from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

PDF_FILE = "docs/RFC791.pdf"
COLLECTION_NAME = "rfc791"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

reader = PdfReader(PDF_FILE)

text = "" 

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

# Split intelligently
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = splitter.split_text(text)

print(f"Total Chunks: {len(chunks)}")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

print(f"Generating embeddings for {len(chunks)} chunks...")

embeddings = embedding_model.encode(chunks)

print("Embeddings generated!")

ids = [f"chunk_{i}" for i in range(len(chunks))]

metadatas = [
    {
        "source": "RFC791.pdf"
    }
    for _ in chunks
]

collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("\nRFC indexed successfully!")
print("Total documents:", collection.count())
