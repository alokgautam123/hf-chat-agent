import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import DOCS_DIR
from services.documents import delete_indexed_document, list_indexed_documents
from services.ingestion import ingest_document
from services.qa import answer_question


app = FastAPI(title="KnowledgeForge")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def render_home(
    request,
    success=None,
    error=None,
    question="",
    qa_result=None,
    status_code=200,
):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "documents": list_indexed_documents(),
            "success": success,
            "error": error,
            "question": question,
            "qa_result": qa_result,
        },
        status_code=status_code,
    )


@app.get("/")
def home(request: Request):
    return render_home(request)


@app.post("/ask")
def ask_question(request: Request, question: str = Form(...)):
    question = question.strip()

    if not question:
        return render_home(
            request,
            error="Enter a question about your indexed documents.",
            status_code=400,
        )

    try:
        result = answer_question(question)
    except Exception:
        return render_home(
            request,
            error="KnowledgeForge could not answer that question.",
            question=question,
            status_code=500,
        )

    return render_home(
        request,
        question=question,
        qa_result=result,
    )


@app.post("/upload")
def upload_document(request: Request, file: UploadFile = File(...)):
    filename = Path(file.filename or "").name

    if not filename or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        return render_home(
            request,
            error="Choose a PDF, Markdown, or text file.",
            status_code=400,
        )

    documents_path = Path(DOCS_DIR)
    documents_path.mkdir(parents=True, exist_ok=True)
    destination = documents_path / filename

    if destination.exists():
        return render_home(
            request,
            error=f"A document named {filename} already exists.",
            status_code=409,
        )

    try:
        with destination.open("wb") as stored_file:
            shutil.copyfileobj(file.file, stored_file)

        ingest_document(destination)
    except Exception:
        if destination.exists():
            destination.unlink()

        return render_home(
            request,
            error=f"KnowledgeForge could not index {filename}.",
            status_code=500,
        )

    return render_home(
        request,
        success=f"{filename} was uploaded and indexed successfully.",
    )


@app.post("/documents/delete")
def delete_document(request: Request, source: str = Form(...)):
    try:
        result = delete_indexed_document(source)
    except Exception:
        return render_home(
            request,
            error=f"KnowledgeForge could not delete {source}.",
            status_code=500,
        )

    if not result["deleted"]:
        return render_home(
            request,
            error=f"{source} was not found.",
            status_code=404,
        )

    return render_home(
        request,
        success=f"{source} was deleted successfully.",
    )
