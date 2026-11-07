
import json
import logging
import re
import shutil
import tempfile
import threading

from functools import lru_cache
from pathlib import Path
from uuid import UUID

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from pydantic import BaseModel, Field

from pipeline import RAGPipeline
from generation.context_builder import ContextBuilder
from generation.generator import AnswerGenerator
from indexing.job_manager import IndexingJobManager

from conversation.memory import conversation_memory
from conversation.query_rewriter import QueryRewriter


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
COLLECTIONS_ROOT = BASE_DIR / "data" / "collections"
MAX_UPLOAD_SIZE = 25 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Atlas RAG API",
    description=(
        "Document question-answering using RAPTOR, "
        "ColBERT, Qwen3, SQLite conversation memory, "
        "and background PDF indexing."
    ),
    version="1.3.0",
)


# --------------------------------------------------
# Background Indexing
# --------------------------------------------------

indexing_jobs = IndexingJobManager(
    max_workers=1,
    collections_root=str(COLLECTIONS_ROOT),
)

collection_delete_lock = threading.Lock()


# --------------------------------------------------
# Request / Response Models
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    collection: str
    session_id: UUID | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class SourceReference(BaseModel):
    citation_id: str
    source: str
    page: int | str | None = None
    leaf_id: str


class QueryResponse(BaseModel):
    question: str
    collection: str
    answer: str
    route: str
    subqueries: list[str]
    sources: list[SourceReference]
    session_id: UUID | None = None
    rewritten_question: str | None = None


class ClearMemoryRequest(BaseModel):
    collection: str
    session_id: UUID


# --------------------------------------------------
# Cached Components
# --------------------------------------------------

@lru_cache(maxsize=3)
def get_pipeline(collection_name: str):
    return RAGPipeline(
        collection_name=collection_name,
        collections_root=str(COLLECTIONS_ROOT),
    )


@lru_cache(maxsize=1)
def get_generator():
    return AnswerGenerator(
        model="qwen3:4b",
        host="http://localhost:11434",
    )


@lru_cache(maxsize=1)
def get_query_rewriter():
    return QueryRewriter(
        model="qwen3:4b",
        host="http://localhost:11434",
    )


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def validate_collection_name(name: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Collection name may contain only letters, "
                "numbers, underscores, and hyphens."
            ),
        )


def get_collection_metadata_path(name: str):
    validate_collection_name(name)

    metadata_path = (
        COLLECTIONS_ROOT / name / "metadata.json"
    )

    if not metadata_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{name}' not found.",
        )

    return metadata_path


def get_safe_filename(filename: str) -> str:
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file has no filename.",
        )

    filename = filename.replace("\\", "/")
    safe_name = filename.rsplit("/", 1)[-1]

    if (
        not safe_name
        or safe_name in {".", ".."}
        or "\x00" in safe_name
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid uploaded filename.",
        )

    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    return safe_name


# --------------------------------------------------
# GET /health
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Atlas RAG API",
        "version": "1.3.0",
        "conversation_memory": True,
        "background_indexing": True,
        "document_deletion": True,
    }


# --------------------------------------------------
# GET /collections
# --------------------------------------------------

@app.get("/collections")
def list_collections():
    collections = []

    if not COLLECTIONS_ROOT.exists():
        return {"collections": collections}

    for metadata_path in sorted(
        COLLECTIONS_ROOT.glob("*/metadata.json")
    ):
        try:
            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as metadata_file:
                metadata = json.load(metadata_file)

            collections.append({
                "name": metadata["collection_name"],
                "filename": metadata["original_filename"],
                "pages": metadata["pages_loaded"],
                "total_nodes": metadata["total_nodes"],
            })

        except (OSError, ValueError, KeyError):
            logger.warning(
                "Skipping invalid metadata: %s",
                metadata_path,
            )

    return {"collections": collections}


# --------------------------------------------------
# POST /query
# --------------------------------------------------

@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_documents(request: QueryRequest):

    get_collection_metadata_path(request.collection)

    try:
        original_question = request.question.strip()

        if not original_question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty.",
            )

        # 1. Load conversation history

        history = []

        if request.session_id is not None:
            history = conversation_memory.get_history(
                session_id=str(request.session_id),
                collection=request.collection,
            )

        # 2. Rewrite follow-up questions

        standalone_question = original_question

        if history:
            standalone_question = (
                get_query_rewriter().rewrite(
                    question=original_question,
                    history=history,
                )
            )

        logger.info(
            "Query: collection=%s, original=%s, rewritten=%s",
            request.collection,
            original_question,
            standalone_question,
        )

        # 3. Run RAG pipeline

        pipeline = get_pipeline(request.collection)

        result = pipeline.run(
            query=standalone_question,
            k=request.top_k,
        )

        # 4. Build evidence context

        context_data = ContextBuilder().build(
            result["results"]
        )

        # 5. Generate grounded answer

        answer = get_generator().generate(
            query=standalone_question,
            context=context_data["context"],
        )

        # 6. Prepare citations

        sources = [
            SourceReference(
                citation_id=source["citation_id"],
                source=source["source"],
                page=source.get("page"),
                leaf_id=source["leaf_id"],
            )
            for source in context_data["sources"]
        ]

        # 7. Save conversation turn

        if request.session_id is not None:
            conversation_memory.add_turn(
                session_id=str(request.session_id),
                collection=request.collection,
                question=original_question,
                answer=answer,
            )

        # 8. Return response

        return QueryResponse(
            question=original_question,
            collection=request.collection,
            answer=answer,
            route=result["route"],
            subqueries=result["subqueries"],
            sources=sources,
            session_id=request.session_id,
            rewritten_question=standalone_question,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("RAG query failed")

        raise HTTPException(
            status_code=500,
            detail="RAG query failed. Check server logs.",
        ) from exc


# --------------------------------------------------
# POST /conversation/clear
# --------------------------------------------------

@app.post("/conversation/clear")
def clear_conversation(request: ClearMemoryRequest):

    get_collection_metadata_path(request.collection)

    conversation_memory.clear(
        session_id=str(request.session_id),
        collection=request.collection,
    )

    return {
        "status": "cleared",
        "collection": request.collection,
        "session_id": str(request.session_id),
    }


# --------------------------------------------------
# GET /jobs/{job_id}
# --------------------------------------------------

@app.get("/jobs/{job_id}")
def get_indexing_job(job_id: str):

    job = indexing_jobs.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Indexing job not found.",
        )

    return job


# --------------------------------------------------
# POST /upload
# --------------------------------------------------

@app.post("/upload", status_code=202)
def upload_pdf(
    collection: str = Form(...),
    file: UploadFile = File(...),
):

    validate_collection_name(collection)

    original_filename = get_safe_filename(
        file.filename
    )

    collection_dir = COLLECTIONS_ROOT / collection

    if collection_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Collection '{collection}' already exists.",
        )

    temp_dir = None
    job_submitted = False

    try:
        temp_dir = Path(
            tempfile.mkdtemp(prefix="atlas_upload_")
        )

        temp_path = temp_dir / original_filename

        uploaded_bytes = 0

        with temp_path.open("wb") as output:
            while True:
                chunk = file.file.read(1024 * 1024)

                if not chunk:
                    break

                uploaded_bytes += len(chunk)

                if uploaded_bytes > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 25 MB limit.",
                    )

                output.write(chunk)

        if uploaded_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        with temp_path.open("rb") as pdf:
            header = pdf.read(5)

        if header != b"%PDF-":
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file signature.",
            )

        logger.info(
            "Queueing ingestion: collection=%s, filename=%s",
            collection,
            original_filename,
        )

        job = indexing_jobs.create_job(
            pdf_path=str(temp_path),
            collection_name=collection,
            cleanup_dir=str(temp_dir),
        )

        job_submitted = True

        return {
            "status": job["status"],
            "job_id": job["job_id"],
            "collection": collection,
            "filename": original_filename,
            "message": (
                "PDF uploaded successfully. "
                "Background indexing started."
            ),
        }

    except HTTPException:
        raise

    except FileExistsError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("PDF upload failed")

        raise HTTPException(
            status_code=500,
            detail="PDF upload failed. Check server logs.",
        ) from exc

    finally:
        file.file.close()

        if temp_dir is not None and not job_submitted:
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )


# --------------------------------------------------
# DELETE /collections/{collection_name}
# --------------------------------------------------

@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str):

    validate_collection_name(collection_name)

    with collection_delete_lock:

        collection_dir = COLLECTIONS_ROOT / collection_name

        # Do not delete collections currently being indexed.
        with indexing_jobs.lock:
            if collection_name in indexing_jobs.active_collections:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "This document is currently being indexed. "
                        "Wait until indexing finishes."
                    ),
                )

        metadata_path = collection_dir / "metadata.json"

        if not metadata_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Collection not found.",
            )

        root = COLLECTIONS_ROOT.resolve()
        target = collection_dir.resolve()

        if target.parent != root or collection_dir.is_symlink():
            raise HTTPException(
                status_code=400,
                detail="Invalid collection path.",
            )

        try:
            # Delete PDF, RAPTOR data, ColBERT index
            # and collection metadata.
            shutil.rmtree(target)

            # Remove cached pipelines.
            get_pipeline.cache_clear()

            # Delete all SQLite messages for this collection.
            conversation_memory.clear_collection(
                collection_name
            )

            logger.info(
                "Deleted collection: %s",
                collection_name,
            )

            return {
                "status": "deleted",
                "collection": collection_name,
            }

        except OSError as exc:
            logger.exception(
                "Failed to delete collection: %s",
                collection_name,
            )

            raise HTTPException(
                status_code=500,
                detail="Failed to delete collection.",
            ) from exc
