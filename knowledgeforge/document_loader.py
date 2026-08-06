from pathlib import Path
from pypdf import PdfReader

from config import DOCS_DIR

def read_pdf_file(file_path):
    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text and page_text.strip():
            pages.append(
                {
                    "text": page_text,
                    "page": page_number
                }
            )

    return pages

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_documents():
    documents = []

    docs_path = Path(DOCS_DIR)

    for file in docs_path.iterdir():
        if not file.is_file():
            continue

        print(f"Loading file: {file.name}")

        if file.suffix in [".md", ".txt"]:
            text = read_text_file(file)

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": file.name,
                        "type": file.suffix.replace(".", "")
                    }
                }
            )
        elif file.suffix == ".pdf":
            pages = read_pdf_file(file)

            for page in pages:
                documents.append(
                    {
                        "text": page["text"],
                        "metadata": {
                            "source": file.name,
                            "type": "pdf",
                            "page": page["page"]
                        }
                    }
                )

    return documents


if __name__ == "__main__":
    docs = load_documents()

    print("\nDocuments loaded:", len(docs))

    for doc in docs:
        print("\nMetadata:")
        print(doc["metadata"])

        print("\nContent:")
        print(doc["text"])
