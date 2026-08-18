import shutil
import secrets
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from config import (
    DOCS_DIR,
    SESSION_COOKIE_SECURE,
    SESSION_MAX_AGE,
    SESSION_SECRET,
)
from services.auth import (
    DuplicateEmailError,
    create_user,
    find_user_by_email,
    find_user_by_id,
    verify_user_password,
)
from services.documents import delete_indexed_document, list_indexed_documents
from services.ingestion import ingest_document
from services.qa import answer_question


if not SESSION_SECRET:
    raise RuntimeError("SESSION_SECRET environment variable is required.")


app = FastAPI(title="KnowledgeForge")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="knowledgeforge_session",
    max_age=SESSION_MAX_AGE,
    same_site="lax",
    https_only=SESSION_COOKIE_SECURE,
)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def get_csrf_token(request):
    token = request.session.get("csrf_token")

    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token

    return token


def valid_csrf_token(request, submitted_token):
    session_token = request.session.get("csrf_token")

    return bool(
        session_token
        and submitted_token
        and secrets.compare_digest(session_token, submitted_token)
    )


def get_current_user(request):
    user_id = request.session.get("user_id")

    if not user_id:
        return None

    user = find_user_by_id(user_id)

    if not user or not user["is_active"]:
        request.session.clear()
        return None

    return {
        "id": user["id"],
        "email": user["email"],
        "is_active": user["is_active"],
    }


def start_authenticated_session(request, user):
    request.session.clear()
    request.session["user_id"] = user["id"]
    request.session["csrf_token"] = secrets.token_urlsafe(32)


def login_redirect():
    return RedirectResponse(url="/login", status_code=303)


def render_auth_page(request, template_name, error=None, email="", status_code=200):
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context={
            "csrf_token": get_csrf_token(request),
            "error": error,
            "email": email,
        },
        status_code=status_code,
    )


def render_home(
    request,
    current_user,
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
            "documents": list_indexed_documents(current_user["id"]),
            "success": success,
            "error": error,
            "question": question,
            "qa_result": qa_result,
            "current_user": current_user,
            "csrf_token": get_csrf_token(request),
        },
        status_code=status_code,
    )


@app.get("/")
def home(request: Request):
    current_user = get_current_user(request)

    if not current_user:
        return login_redirect()

    return render_home(request, current_user)


@app.get("/register")
def register_page(request: Request):
    if get_current_user(request):
        return RedirectResponse(url="/", status_code=303)

    return render_auth_page(request, "register.html")


@app.post("/register")
def register(
    request: Request,
    email: str = Form(""),
    password: str = Form(""),
    csrf_token: str = Form(""),
):
    if get_current_user(request):
        return RedirectResponse(url="/", status_code=303)

    if not valid_csrf_token(request, csrf_token):
        return render_auth_page(
            request,
            "register.html",
            error="Your form expired. Please try again.",
            email=email,
            status_code=400,
        )

    try:
        user = create_user(email, password)
    except DuplicateEmailError:
        return render_auth_page(
            request,
            "register.html",
            error="An account with that email already exists.",
            email=email,
            status_code=409,
        )
    except ValueError as error:
        return render_auth_page(
            request,
            "register.html",
            error=str(error),
            email=email,
            status_code=400,
        )

    start_authenticated_session(request, user)
    return RedirectResponse(url="/", status_code=303)


@app.get("/login")
def login_page(request: Request):
    if get_current_user(request):
        return RedirectResponse(url="/", status_code=303)

    return render_auth_page(request, "login.html")


@app.post("/login")
def login(
    request: Request,
    email: str = Form(""),
    password: str = Form(""),
    csrf_token: str = Form(""),
):
    if get_current_user(request):
        return RedirectResponse(url="/", status_code=303)

    if not valid_csrf_token(request, csrf_token):
        return render_auth_page(
            request,
            "login.html",
            error="Your form expired. Please try again.",
            email=email,
            status_code=400,
        )

    user = find_user_by_email(email)

    if not verify_user_password(user, password):
        return render_auth_page(
            request,
            "login.html",
            error="Invalid email or password.",
            email=email,
            status_code=401,
        )

    start_authenticated_session(request, user)
    return RedirectResponse(url="/", status_code=303)


@app.post("/logout")
def logout(request: Request, csrf_token: str = Form("")):
    current_user = get_current_user(request)

    if not current_user:
        return login_redirect()

    if not valid_csrf_token(request, csrf_token):
        return render_home(
            request,
            current_user,
            error="Your form expired. Please try again.",
            status_code=400,
        )

    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.post("/ask")
def ask_question(
    request: Request,
    question: str = Form(""),
    csrf_token: str = Form(""),
):
    current_user = get_current_user(request)

    if not current_user:
        return login_redirect()

    if not valid_csrf_token(request, csrf_token):
        return render_home(
            request,
            current_user,
            error="Your form expired. Please try again.",
            question=question,
            status_code=400,
        )

    question = question.strip()

    if not question:
        return render_home(
            request,
            current_user,
            error="Enter a question about your indexed documents.",
            status_code=400,
        )

    try:
        result = answer_question(question, current_user["id"])
    except Exception:
        return render_home(
            request,
            current_user,
            error="KnowledgeForge could not answer that question.",
            question=question,
            status_code=500,
        )

    return render_home(
        request,
        current_user,
        question=question,
        qa_result=result,
    )


@app.post("/upload")
def upload_document(
    request: Request,
    file: UploadFile | None = File(None),
    csrf_token: str = Form(""),
):
    current_user = get_current_user(request)

    if not current_user:
        return login_redirect()

    if not valid_csrf_token(request, csrf_token):
        return render_home(
            request,
            current_user,
            error="Your form expired. Please try again.",
            status_code=400,
        )

    if file is None:
        return render_home(
            request,
            current_user,
            error="Choose a PDF, Markdown, or text file.",
            status_code=400,
        )

    filename = Path(file.filename or "").name

    if not filename or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        return render_home(
            request,
            current_user,
            error="Choose a PDF, Markdown, or text file.",
            status_code=400,
        )

    documents_path = Path(DOCS_DIR) / current_user["id"]
    documents_path.mkdir(parents=True, exist_ok=True)
    destination = documents_path / filename

    if destination.exists():
        return render_home(
            request,
            current_user,
            error=f"A document named {filename} already exists.",
            status_code=409,
        )

    try:
        with destination.open("wb") as stored_file:
            shutil.copyfileobj(file.file, stored_file)

        ingest_document(destination, current_user["id"])
    except Exception:
        if destination.exists():
            destination.unlink()

        return render_home(
            request,
            current_user,
            error=f"KnowledgeForge could not index {filename}.",
            status_code=500,
        )

    return render_home(
        request,
        current_user,
        success=f"{filename} was uploaded and indexed successfully.",
    )


@app.post("/documents/delete")
def delete_document(
    request: Request,
    source: str = Form(""),
    csrf_token: str = Form(""),
):
    current_user = get_current_user(request)

    if not current_user:
        return login_redirect()

    if not valid_csrf_token(request, csrf_token):
        return render_home(
            request,
            current_user,
            error="Your form expired. Please try again.",
            status_code=400,
        )

    try:
        result = delete_indexed_document(source, current_user["id"])
    except Exception:
        return render_home(
            request,
            current_user,
            error=f"KnowledgeForge could not delete {source}.",
            status_code=500,
        )

    if not result["deleted"]:
        return render_home(
            request,
            current_user,
            error=f"{source} was not found.",
            status_code=404,
        )

    return render_home(
        request,
        current_user,
        success=f"{source} was deleted successfully.",
    )
