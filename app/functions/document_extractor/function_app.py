"""
Azure Function: Document Extractor
Custom skill for Azure AI Search that extracts and processes document content.
"""

import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

import azure.functions as func
from azure.core.exceptions import HttpResponseError
from azure.identity.aio import ManagedIdentityCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from azure.storage.blob.aio import BlobServiceClient

from prepdocslib.blobmanager import BlobManager
from prepdocslib.fileprocessor import FileProcessor
from prepdocslib.page import Page
from prepdocslib.servicesetup import (
    build_file_processors,
    select_processor_for_filename,
)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

logger = logging.getLogger(__name__)


@dataclass
class GlobalSettings:
    file_processors: dict[str, FileProcessor]
    azure_credential: ManagedIdentityCredential
    blob_service_client: BlobServiceClient | None
    blob_manager: BlobManager | None


settings: GlobalSettings | None = None


def configure_global_settings():
    global settings

    # Environment configuration
    use_local_pdf_parser = os.getenv("USE_LOCAL_PDF_PARSER", "false").lower() == "true"
    use_local_html_parser = os.getenv("USE_LOCAL_HTML_PARSER", "false").lower() == "true"
    use_multimodal = os.getenv("USE_MULTIMODAL", "false").lower() == "true"
    document_intelligence_service = os.getenv("AZURE_DOCUMENTINTELLIGENCE_SERVICE")
    storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
    storage_container = os.getenv("AZURE_STORAGE_CONTAINER", "content")

    # Single shared managed identity credential
    if AZURE_CLIENT_ID := os.getenv("AZURE_CLIENT_ID"):
        logger.info("Using Managed Identity with client ID: %s", AZURE_CLIENT_ID)
        azure_credential = ManagedIdentityCredential(client_id=AZURE_CLIENT_ID)
    else:
        logger.info("Using default Managed Identity without client ID")
        azure_credential = ManagedIdentityCredential()

    # Build file processors dict for parser selection
    file_processors = build_file_processors(
        azure_credential=azure_credential,
        document_intelligence_service=document_intelligence_service,
        document_intelligence_key=None,
        use_local_pdf_parser=use_local_pdf_parser,
        use_local_html_parser=use_local_html_parser,
        process_figures=use_multimodal,
    )

    # Create blob service client
    blob_service_client = None
    blob_manager = None
    if storage_account:
        blob_endpoint = f"https://{storage_account}.blob.core.windows.net"
        blob_service_client = BlobServiceClient(account_url=blob_endpoint, credential=azure_credential)
        logger.info("Initialized BlobServiceClient for %s", storage_account)
        
        # Initialize blob manager for uploading extracted images
        blob_manager = BlobManager(
            endpoint=blob_endpoint,
            container=storage_container,
            account=storage_account,
            credential=azure_credential
        )
        logger.info("Initialized BlobManager for container %s", storage_container)

    settings = GlobalSettings(
        file_processors=file_processors,
        azure_credential=azure_credential,
        blob_service_client=blob_service_client,
        blob_manager=blob_manager,
    )


@app.function_name(name="extract")
@app.route(route="extract", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
async def extract_document(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Search Custom Skill: Extract document content

    Input format (single record; file data only):
    # https://learn.microsoft.com/azure/search/cognitive-search-skill-document-intelligence-layout#skill-inputs
    {
        "values": [
            {
                "recordId": "1",
                "data": {
                    // Base64 encoded file (skillset must enable file data)
                    "file_data": {
                        "$type": "file",
                        "data": "base64..."
                    },
                    // Optional
                    "file_name": "doc.pdf"
                }
            }
        ]
    }

    Output format (snake_case only):
    {
        "values": [
            {
                "recordId": "1",
                "data": {
                    "pages": [
                        {"page_num": 0, "text": "Page 1 text", "figure_ids": ["fig1"]},
                        {"page_num": 1, "text": "Page 2 text", "figure_ids": []}
                    ],
                    "figures": [
                        {
                            "figure_id": "fig1",
                            "page_num": 0,
                            "document_file_name": "doc.pdf",
                            "filename": "fig1.png",
                            "mime_type": "image/png",
                            "bytes_base64": "...",
                            "bbox": [100,150,300,400],
                            "title": "Figure Title",
                            "placeholder": "<figure id=\"fig1\"></figure>"
                        }
                    ]
                },
                "errors": [],
                "warnings": []
            }
        ]
    }
    """
    if settings is None:
        return func.HttpResponse(
            json.dumps({"error": "Settings not initialized"}),
            mimetype="application/json",
            status_code=500,
        )

    try:
        # Parse custom skill input
        req_body = req.get_json()
        input_values = req_body.get("values", [])

        if len(input_values) != 1:
            raise ValueError("document_extractor expects exactly one record per request, set batchSize to 1.")

        input_record = input_values[0]
        record_id = input_record.get("recordId", "")
        data = input_record.get("data", {})

        try:
            result = await process_document(data)
            output_values = [
                {
                    "recordId": record_id,
                    "data": result,
                    "errors": [],
                    "warnings": [],
                }
            ]
        except Exception as e:
            logger.error("Error processing record %s: %s", record_id, str(e), exc_info=True)
            output_values = [
                {
                    "recordId": record_id,
                    "data": {},
                    "errors": [{"message": str(e)}],
                    "warnings": [],
                }
            ]

        return func.HttpResponse(json.dumps({"values": output_values}), mimetype="application/json", status_code=200)

    except Exception as e:
        logger.error("Fatal error in extract_document: %s", str(e), exc_info=True)
        return func.HttpResponse(json.dumps({"error": str(e)}), mimetype="application/json", status_code=500)


async def process_document(data: dict[str, Any]) -> dict[str, Any]:
    """
    Process a single document: download, parse, extract figures, upload images

    Args:
        data: Input data with file_data (small files) or metadata_storage_path (large files)

    Returns:
        Dictionary with pages and figures (with URLs if uploaded, base64 if not)
    """
    # Log available fields for debugging
    logger.info("Available input fields: %s", list(data.keys()))
    
    # Try file_data first (for small files), fall back to blob URL (for large files)
    if "file_data" in data and data.get("file_data", {}).get("data"):
        document_stream, file_name, content_type = get_document_stream_filedata(data)
    elif "metadata_storage_path" in data:
        document_stream, file_name, content_type = await get_document_stream_from_blob_url(data, "metadata_storage_path")
    elif "normalized_images" in data and "metadata_storage_path" in data.get("normalized_images", {}):
        # Handle Azure Search normalized_images format
        document_stream, file_name, content_type = await get_document_stream_from_blob_url(data["normalized_images"], "metadata_storage_path")
    elif any(key.endswith("_path") for key in data.keys()):
        # Try to find any field ending with _path that might contain blob URL
        path_key = next((k for k in data.keys() if k.endswith("_path")), None)
        logger.info("Found alternative path field: %s", path_key)
        document_stream, file_name, content_type = await get_document_stream_from_blob_url(data, path_key)
    else:
        # Provide detailed error message with available fields
        available_fields = ", ".join(data.keys()) if data else "none"
        raise ValueError(
            f"Input must contain either 'file_data' with base64 data or 'metadata_storage_path' with blob URL. "
            f"For files larger than 16MB, the indexer cannot send file_data inline. "
            f"Available fields in input: {available_fields}. "
            f"Update your skillset to pass 'metadata_storage_path' or configure file data enrichment."
        )
    
    file_size_mb = len(document_stream.getvalue()) / (1024 * 1024)
    logger.info("Processing document: %s (%.2f MB)", file_name, file_size_mb)

    # Get parser from file_processors dict based on file extension
    file_processor = select_processor_for_filename(file_name, settings.file_processors)
    parser = file_processor.parser

    pages: list[Page] = []
    try:
        document_stream.seek(0)
        page_count = 0
        async for page in parser.parse(content=document_stream):
            pages.append(page)
            page_count += 1
            if page_count % 50 == 0:
                logger.info("Processed %d pages...", page_count)
    except HttpResponseError as exc:
        raise ValueError(f"Parser failed for {file_name}: {exc.message}") from exc
    except Exception as exc:
        logger.error("Parse error after %d pages: %s", len(pages), str(exc), exc_info=True)
        raise ValueError(f"Parser failed for {file_name}: {str(exc)}") from exc
    finally:
        document_stream.close()

    logger.info("Successfully parsed %d pages from %s", len(pages), file_name)
    
    # Upload extracted images to blob storage if blob manager is configured
    upload_images = settings.blob_manager is not None
    components = await build_document_components(file_name, pages, upload_images)
    return components


async def get_document_stream_from_blob_url(data: dict[str, Any], url_field: str = "metadata_storage_path") -> tuple[io.BytesIO, str, str]:
    """Download document from blob storage using metadata_storage_path or other URL field."""
    if settings.blob_service_client is None:
        raise ValueError("Blob storage not configured. Set AZURE_STORAGE_ACCOUNT environment variable.")
    
    blob_url = data.get(url_field)
    if not blob_url:
        raise ValueError(f"{url_field} not found in input data")
    
    logger.info("Downloading from blob URL field '%s': %s", url_field, blob_url)
    
    # Parse blob URL: https://account.blob.core.windows.net/container/path/file.pdf
    # Handle both full URLs and SAS URLs
    url_parts = blob_url.split("?")[0].split("/")  # Remove SAS token if present
    
    if len(url_parts) < 5:
        raise ValueError(f"Invalid blob URL format: {blob_url}")
    
    container_name = url_parts[3]
    # URL decode the blob name to handle spaces and special characters
    blob_name_encoded = "/".join(url_parts[4:])
    blob_name = unquote(blob_name_encoded)
    
    # Try multiple possible name fields
    file_name = (
        data.get("metadata_storage_name") or 
        data.get("file_name") or 
        data.get("fileName") or 
        blob_name.split("/")[-1]
    )
    
    logger.info("Downloading from blob: container='%s', blob='%s' (decoded from '%s')", container_name, blob_name, blob_name_encoded)
    
    blob_client = settings.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    try:
        download_stream = await blob_client.download_blob()
        blob_bytes = await download_stream.readall()
        logger.info("Downloaded %d bytes for file: %s", len(blob_bytes), file_name)
        
        stream = io.BytesIO(blob_bytes)
        stream.name = file_name
        return stream, file_name, "application/pdf"
    except Exception as e:
        logger.error("Failed to download blob %s/%s: %s", container_name, blob_name, str(e), exc_info=True)
        raise ValueError(f"Failed to download from blob storage: {str(e)}") from e


def get_document_stream_filedata(data: dict[str, Any]) -> tuple[io.BytesIO, str, str]:
    """Return a BytesIO stream for file_data input only (skillset must send file bytes)."""
    file_payload = data.get("file_data", {})
    
    if not file_payload:
        raise ValueError("file_data field is empty or missing")
    
    encoded = file_payload.get("data")
    if not encoded:
        raise ValueError(
            "file_data payload missing base64 data. "
            "This typically means the file exceeds the 16MB limit for inline data. "
            "Update your skillset to pass 'metadata_storage_path' instead."
        )
    
    try:
        document_bytes = base64.b64decode(encoded)
        logger.info("Decoded %d bytes from base64", len(document_bytes))
    except Exception as e:
        raise ValueError(f"Failed to decode base64 data: {str(e)}") from e
    
    file_name = data.get("file_name") or data.get("fileName") or file_payload.get("name") or "document"
    content_type = data.get("contentType") or file_payload.get("contentType") or "application/octet-stream"
    stream = io.BytesIO(document_bytes)
    stream.name = file_name
    return stream, file_name, content_type


async def build_document_components(file_name: str, pages: list[Page], upload_images: bool = False) -> dict[str, Any]:
    """Build document components with optional image upload to blob storage."""
    page_entries: list[dict[str, Any]] = []
    figure_entries: list[dict[str, Any]] = []

    for page in pages:
        page_text = page.text or ""
        figure_ids_on_page: list[str] = []
        
        if page.images:
            for image in page.images:
                figure_ids_on_page.append(image.figure_id)
                
                # Upload image to blob storage if enabled
                if upload_images and settings.blob_manager:
                    try:
                        # Upload the image bytes to blob storage
                        blob_path = f"images/{file_name}/{image.figure_id}.png"
                        await settings.blob_manager.upload_blob(
                            blob_path,
                            image.get_image_bytes()
                        )
                        # Add URL to the figure entry instead of base64
                        figure_payload = image.to_skill_payload(file_name)
                        figure_payload["url"] = f"{settings.blob_manager.endpoint}/{settings.blob_manager.container}/{blob_path}"
                        # Remove base64 data to reduce response size
                        figure_payload.pop("bytes_base64", None)
                        figure_entries.append(figure_payload)
                        logger.info("Uploaded image %s to blob storage", image.figure_id)
                    except Exception as e:
                        logger.warning("Failed to upload image %s: %s", image.figure_id, str(e))
                        # Fall back to base64 if upload fails
                        figure_entries.append(image.to_skill_payload(file_name))
                else:
                    # Return base64 encoded image inline
                    figure_entries.append(image.to_skill_payload(file_name))

        page_entries.append(
            {
                "page_num": page.page_num,
                "text": page_text,
                "figure_ids": figure_ids_on_page,
            }
        )

    return {
        "file_name": file_name,
        "pages": page_entries,
        "figures": figure_entries,
    }


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
