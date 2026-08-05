from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE
from config import CHUNK_OVERLAP


_splitter = None


def get_text_splitter():
    global _splitter

    if _splitter is None:
        print("Loading text splitter...")

        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    return _splitter


def create_chunks(documents):
    splitter = get_text_splitter()

    chunks = []

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        split_text = splitter.split_text(text)

        for chunk in split_text:
            chunks.append(
                {
                    "text": chunk,
                    "metadata": metadata.copy()
                }
            )

    return chunks


if __name__ == "__main__":
    documents = [
        {
            "text": (
                "KnowledgeForge is an AI knowledge assistant. "
                * 100
            ),
            "metadata": {
                "source": "test.md"
            }
        }
    ]

    chunks = create_chunks(documents)

    print(f"Created {len(chunks)} chunks")
