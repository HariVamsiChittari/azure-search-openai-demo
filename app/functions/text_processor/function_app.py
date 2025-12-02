"""Azure Function: Text Processor.
Custom skill for Azure AI Search that merges page text with figure metadata, splits into chunks, and computes embeddings.
"""

import io
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import azure.functions as func
from azure.identity.aio import ManagedIdentityCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

from prepdocslib.blobmanager import BlobManager
from prepdocslib.embeddings import OpenAIEmbeddings
from prepdocslib.fileprocessor import FileProcessor
from prepdocslib.listfilestrategy import File
from prepdocslib.page import ImageOnPage, Page
from prepdocslib.servicesetup import (
    OpenAIHost,
    build_file_processors,
    select_processor_for_filename,
    setup_embeddings_service,
    setup_openai_client,
)
from prepdocslib.textprocessor import process_text

# Mark the function as anonymous since we are protecting it with built-in auth instead
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

logger = logging.getLogger(__name__)


@dataclass
class GlobalSettings:
    use_vectors: bool
    use_multimodal: bool
    embedding_dimensions: int
    file_processors: dict[str, FileProcessor]
    embedding_service: OpenAIEmbeddings | None


settings: GlobalSettings | None = None


def configure_global_settings():
    global settings

    # Environment configuration
    use_vectors = os.getenv("USE_VECTORS", "true").lower() == "true"
    use_multimodal = os.getenv("USE_MULTIMODAL", "false").lower() == "true"
    embedding_dimensions = int(os.getenv("AZURE_OPENAI_EMB_DIMENSIONS", "3072"))

    # Conditionally required (based on feature flags)
    openai_host_str = os.getenv("OPENAI_HOST", "azure")
    azure_openai_service = os.getenv("AZURE_OPENAI_SERVICE")
    azure_openai_custom_url = os.getenv("AZURE_OPENAI_CUSTOM_URL")
    azure_openai_emb_deployment = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")
    azure_openai_emb_model_name = os.getenv("AZURE_OPENAI_EMB_MODEL_NAME", "text-embedding-3-large")
    document_intelligence_service = os.getenv("AZURE_DOCUMENTINTELLIGENCE_SERVICE")

    # Single shared managed identity credential
    if AZURE_CLIENT_ID := os.getenv("AZURE_CLIENT_ID"):
        logger.info("Using Managed Identity with client ID: %s", AZURE_CLIENT_ID)
        azure_credential = ManagedIdentityCredential(client_id=AZURE_CLIENT_ID)
    else:
        logger.info("Using default Managed Identity without client ID")
        azure_credential = ManagedIdentityCredential()

    # Build file processors to get correct splitter for each file type
    file_processors = build_file_processors(
        azure_credential=azure_credential,
        document_intelligence_service=document_intelligence_service,
        document_intelligence_key=None,
        use_local_pdf_parser=False,
        use_local_html_parser=False,
        process_figures=use_multimodal,
    )

    # Embedding service (optional)
    embedding_service = None
    if use_vectors:
        if (azure_openai_service or azure_openai_custom_url) and (
            azure_openai_emb_deployment and azure_openai_emb_model_name
        ):
            openai_host = OpenAIHost(openai_host_str)
            openai_client, azure_openai_endpoint = setup_openai_client(
                openai_host=openai_host,
                azure_credential=azure_credential,
                azure_openai_service=azure_openai_service,
                azure_openai_custom_url=azure_openai_custom_url,
            )
            embedding_service = setup_embeddings_service(
                openai_host,
                openai_client,
                emb_model_name=azure_openai_emb_model_name,
                emb_model_dimensions=embedding_dimensions,
                azure_openai_deployment=azure_openai_emb_deployment,
                azure_openai_endpoint=azure_openai_endpoint,
            )
        else:
            logger.warning("USE_VECTORS is true but embedding configuration incomplete; embeddings disabled")

    settings = GlobalSettings(
        use_vectors=use_vectors,
        use_multimodal=use_multimodal,
        embedding_dimensions=embedding_dimensions,
        file_processors=file_processors,
        embedding_service=embedding_service,
    )


@app.function_name(name="process_text")
@app.route(route="process", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
async def process_text_entry(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Search custom skill entry point for chunking and embeddings."""
    start_time = time.time()
    
    if settings is None:
        logger.error("[METRICS] function=text_processor | operation=error | error_type=initialization_error")
        return func.HttpResponse(
            json.dumps({"error": "Settings not initialized"}),
            mimetype="application/json",
            status_code=500,
        )

    try:
        payload = req.get_json()
    except ValueError as exc:
        logger.error("[METRICS] function=text_processor | operation=error | error_type=invalid_json | error=%s", exc)
        return func.HttpResponse(
            json.dumps({"error": "Request body must be valid JSON"}),
            mimetype="application/json",
            status_code=400,
        )

    values = payload.get("values", [])
    total_records = len(values)
    successful_records = 0
    failed_records = 0
    total_chunks = 0
    
    logger.info(
        "[METRICS] function=text_processor | operation=batch_start | record_count=%d | use_vectors=%s | use_multimodal=%s",
        total_records, settings.use_vectors, settings.use_multimodal
    )
    
    output_values: list[dict[str, Any]] = []

    for record in values:
        record_id = record.get("recordId", "")
        data = record.get("data", {})
        record_start_time = time.time()
        
        try:
            chunks = await process_document(data)
            record_duration = time.time() - record_start_time
            chunk_count = len(chunks)
            total_chunks += chunk_count
            successful_records += 1
            
            logger.info(
                "[METRICS] function=text_processor | operation=record_complete | record_id=%s | "
                "chunk_count=%d | duration_sec=%.2f | status=success",
                record_id, chunk_count, record_duration
            )
            
            output_values.append(
                {
                    "recordId": record_id,
                    "data": {"chunks": chunks},
                    "errors": [],
                    "warnings": [],
                }
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            record_duration = time.time() - record_start_time
            failed_records += 1
            
            logger.error(
                "[METRICS] function=text_processor | operation=record_complete | record_id=%s | "
                "error_type=%s | error_message=%s | duration_sec=%.2f | status=failed",
                record_id, type(exc).__name__, str(exc), record_duration, exc_info=True
            )
            
            output_values.append(
                {
                    "recordId": record_id,
                    "data": {},
                    "errors": [{"message": str(exc)}],
                    "warnings": [],
                }
            )

    total_duration = time.time() - start_time
    success_rate = (successful_records / total_records * 100) if total_records > 0 else 0
    
    logger.info(
        "[METRICS] function=text_processor | operation=batch_complete | total_records=%d | "
        "successful_records=%d | failed_records=%d | total_chunks=%d | duration_sec=%.2f | success_rate=%.1f",
        total_records, successful_records, failed_records, total_chunks, total_duration, success_rate
    )

    return func.HttpResponse(
        json.dumps({"values": output_values}),
        mimetype="application/json",
        status_code=200,
    )


async def process_document(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Combine figures with page text, split into chunks, and (optionally) embed.

    Parameters
    ----------
    data: dict[str, Any]
        Skill payload containing consolidated_document with file metadata, pages, and figures.

    Returns
    -------
    list[dict[str, Any]]
        Chunk dictionaries ready for downstream indexing.
    """

    # Extract consolidated_document object from Shaper skill
    consolidated_doc = data.get("consolidated_document", data)

    file_name = consolidated_doc.get("file_name", "document")
    storage_url = consolidated_doc.get("storageUrl") or consolidated_doc.get("metadata_storage_path") or file_name
    pages_input = consolidated_doc.get("pages", [])  # [{page_num, text, figure_ids}]
    figures_input = consolidated_doc.get("figures", [])  # serialized skill payload

    figures_by_id = {figure["figure_id"]: figure for figure in figures_input}

    logger.info(
        "[METRICS] function=text_processor | operation=document_start | file=%s | "
        "page_count=%d | figure_count=%d",
        file_name, len(pages_input), len(figures_input)
    )

    # Build Page objects with placeholders intact (figure markup will be injected by combine_text_with_figures())
    pages: list[Page] = []
    offset = 0
    for page_entry in pages_input:
        # Zero-based page numbering: pages emitted by extractor already zero-based
        page_num = int(page_entry.get("page_num", len(pages)))
        page_text = page_entry.get("text", "")
        page_obj = Page(page_num=page_num, offset=offset, text=page_text)
        offset += len(page_text)

        # Construct ImageOnPage objects from figureIds list
        figure_ids: list[str] = page_entry.get("figure_ids", [])
        for fid in figure_ids:
            figure_payload = figures_by_id.get(fid)
            if not figure_payload:
                logger.warning("Figure ID %s not found in figures metadata for page %d", fid, page_num)
                continue
            try:
                image_on_page, _ = ImageOnPage.from_skill_payload(figure_payload)
                page_obj.images.append(image_on_page)
            except Exception as exc:
                logger.error("Failed to deserialize figure %s: %s", fid, exc, exc_info=True)
        pages.append(page_obj)

    if not pages:
        logger.info(
            "[METRICS] function=text_processor | operation=document_complete | file=%s | "
            "output_chunks=0 | reason=no_pages",
            file_name
        )
        return []

    # Create a lightweight File wrapper required by process_text
    dummy_stream = io.BytesIO(b"")
    dummy_stream.name = file_name
    file_wrapper = File(content=dummy_stream)

    # Get the appropriate splitter for this file type
    file_processor = select_processor_for_filename(file_name, settings.file_processors)
    splitter = file_processor.splitter

    sections = process_text(pages, file_wrapper, splitter, category=None)
    if not sections:
        logger.info(
            "[METRICS] function=text_processor | operation=document_complete | file=%s | "
            "output_chunks=0 | reason=no_sections",
            file_name
        )
        return []

    # Generate embeddings for section texts
    chunk_texts = [s.chunk.text for s in sections]
    embeddings: list[list[float]] | None = None
    if settings.use_vectors and chunk_texts:
        if settings.embedding_service:
            embedding_start_time = time.time()
            embeddings = await settings.embedding_service.create_embeddings(chunk_texts)
            embedding_duration = time.time() - embedding_start_time
            
            logger.info(
                "[METRICS] function=text_processor | operation=embeddings_complete | file=%s | "
                "embedding_count=%d | duration_sec=%.2f",
                file_name, len(embeddings) if embeddings else 0, embedding_duration
            )
        else:
            logger.warning(
                "[METRICS] function=text_processor | operation=embeddings_skipped | file=%s | reason=service_not_initialized",
                file_name
            )

    # Use the same id base generation as local ingestion pipeline for parity
    normalized_id = file_wrapper.filename_to_id()
    outputs: list[dict[str, Any]] = []
    skipped_embeddings = 0
    
    for idx, section in enumerate(sections):
        content = section.chunk.text.strip()
        if not content:
            continue
        embedding_vec = embeddings[idx] if embeddings else None
        image_refs: list[dict[str, Any]] = []
        for image in section.chunk.images:
            ref = {
                "url": image.url or "",
                "description": image.description or "",
                "boundingbox": list(image.bbox),
            }
            if settings.use_multimodal and image.embedding is not None:
                ref["embedding"] = image.embedding
            image_refs.append(ref)
        chunk_entry: dict[str, Any] = {
            "id": f"{normalized_id}-{idx:04d}",
            "content": content,
            "sourcepage": BlobManager.sourcepage_from_file_page(file_name, section.chunk.page_num),
            "sourcefile": file_name,
            "parent_id": storage_url,
            **({"images": image_refs} if image_refs else {}),
        }

        if embedding_vec is not None:
            if len(embedding_vec) == settings.embedding_dimensions:
                chunk_entry["embedding"] = embedding_vec
            else:
                skipped_embeddings += 1
                logger.warning(
                    "[METRICS] function=text_processor | operation=embedding_dimension_mismatch | file=%s | "
                    "chunk_index=%d | expected_dimensions=%d | actual_dimensions=%d",
                    file_name, idx, settings.embedding_dimensions, len(embedding_vec)
                )
        elif settings.use_vectors:
            skipped_embeddings += 1
            logger.warning(
                "[METRICS] function=text_processor | operation=embedding_missing | file=%s | chunk_index=%d",
                file_name, idx
            )

        outputs.append(chunk_entry)

    image_ref_count = sum(len(chunk.get("images", [])) for chunk in outputs)
    
    logger.info(
        "[METRICS] function=text_processor | operation=document_complete | file=%s | "
        "total_sections=%d | output_chunks=%d | image_references=%d | skipped_embeddings=%d",
        file_name, len(sections), len(outputs), image_ref_count, skipped_embeddings
    )

    return outputs


# Initialize settings and configure monitoring
if os.environ.get("PYTEST_CURRENT_TEST") is None:
    # Configure Azure Monitor telemetry
    if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        logger.info("APPLICATIONINSIGHTS_CONNECTION_STRING is set, enabling Azure Monitor")
        configure_azure_monitor(
            instrumentation_options={
                "django": {"enabled": False},
                "psycopg2": {"enabled": False},
                "fastapi": {"enabled": False},
            }
        )
        # This tracks HTTP requests made by aiohttp:
        AioHttpClientInstrumentor().instrument()
        # This tracks HTTP requests made by httpx:
        HTTPXClientInstrumentor().instrument()
        # This tracks OpenAI SDK requests:
        OpenAIInstrumentor().instrument()

    # Log levels should be one of https://docs.python.org/3/library/logging.html#logging-levels
    # Set root level to WARNING to avoid seeing overly verbose logs from SDKS
    logging.basicConfig(level=logging.WARNING)
    # Set our own logger levels to INFO by default
    app_level = os.getenv("APP_LOG_LEVEL", "INFO")
    logger.setLevel(app_level)
    logging.getLogger("scripts").setLevel(app_level)

    try:
        configure_global_settings()
    except KeyError as e:
        logger.warning("Could not initialize settings at module load time: %s", e)
