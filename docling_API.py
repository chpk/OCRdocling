# main.py
import os
import openai
from pathlib import Path
import logging
import time
import base64
import tempfile
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # Optional: For frontend integration
from pydantic import BaseModel
import uvicorn

# Docling imports
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
_log = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="OCR and Document Structuring API",
    description="Processes an image, performs OCR, and structures the content using a multimodal LLM.",
    version="1.0.0",
)

# Optional: Add CORS middleware if your frontend is on a different domain
# app.add_middleware(
# CORSMiddleware,
# allow_origins=["*"], # Allows all origins
# allow_credentials=True,
# allow_methods=["*"], # Allows all methods
# allow_headers=["*"], # Allows all headers
# )


# --- Helper Functions (Adapted and Enhanced for API) ---

def encode_image_to_base64(image_path: Path) -> str:
    """Reads an image file and encodes it as a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        _log.error(f"Image file not found at {image_path} for Base64 encoding.")
        # This is an internal error, as the file should exist at this point
        raise HTTPException(status_code=500, detail=f"Internal error: Could not find temporary image for encoding.")
    except Exception as e:
        _log.error(f"Error encoding image {image_path} to Base64: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: Could not encode image.")


async def structure_with_multimodal_llm(raw_text: str, image_path: Path, image_original_filename: str) -> str:
    """
    Sends raw OCR text AND the original image to GPT-4o to be cleaned,
    structured, and formatted, using the image as a visual reference.
    Raises HTTPException on failure.
    """
    _log.info(f"Sending OCR text and source image '{image_original_filename}' to OpenAI GPT-4o for structuring...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _log.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        raise HTTPException(status_code=500, detail="OpenAI API key not configured on the server. Please contact support.")

    client = openai.OpenAI(api_key=api_key)

    base64_image = encode_image_to_base64(image_path) # Can raise HTTPException

    file_extension = Path(image_original_filename).suffix.lower()
    mime_type_map = {
        ".jpeg": "image/jpeg", ".jpg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"
    }
    mime_type = mime_type_map.get(file_extension)

    if not mime_type:
        _log.warning(f"Unsupported image type for LLM based on extension: {file_extension}. Attempting image/jpeg.")
        # Fallback or error if a specific list of supported types is enforced
        # For now, we let OpenAI decide if it can handle it with a common default
        mime_type = "image/jpeg"
        # Alternatively, raise HTTPException(status_code=400, detail=f"Unsupported image file type: {file_extension}")

    prompt_text = f"""
    You are an expert document formatter. Your task is to "beautify" the raw text extracted by an OCR tool by using the provided image as the absolute source of truth.

    The attached image is the original document. The following is the raw text extracted by OCR:
    ---
    {raw_text}
    ---

    Your instructions are:
    1.  **Use the image as the ground truth.** Cross-reference the OCR text with the image to ensure accuracy.
    2.  **Format the output as clean Markdown.** Use headings, bold text, and lists to match the visual structure in the image.
    3.  **Accurately represent tables.** If you see a table in the image, format it as a proper Markdown table.
    4.  **Do NOT hallucinate or add information.** Your job is to format the content that is VISUALLY PRESENT in the image. If the OCR text has a minor error (e.g., 'helo' instead of 'hello'), correct it based on the image. If a section is unreadable in the image, write `[Illegible]`. Do not invent data.
    5.  **Return ONLY the final Markdown content.** Do not include any other conversation or commentary.
    """

    try:
        start_time = time.time()
        # For a fully async app, use openai.AsyncOpenAI() and await the call
        response = client.chat.completions.create(
            model="gpt-4o", # Ensure this model is available and appropriate
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=2000 # Adjust as needed
        )
        end_time = time.time() - start_time
        _log.info(f"Multimodal LLM processing for '{image_original_filename}' complete in {end_time:.2f} seconds.")

        structured_content = response.choices[0].message.content
        if not structured_content or structured_content.isspace():
            _log.error(f"LLM returned empty or whitespace content for '{image_original_filename}'.")
            raise HTTPException(status_code=500, detail="LLM processing returned empty content.")
        return structured_content

    except openai.APIConnectionError as e:
        _log.error(f"OpenAI API request failed to connect for '{image_original_filename}': {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Failed to connect to OpenAI API: {e}")
    except openai.RateLimitError as e:
        _log.error(f"OpenAI API request exceeded rate limit for '{image_original_filename}': {e}", exc_info=True)
        raise HTTPException(status_code=429, detail=f"OpenAI API rate limit exceeded: {e}")
    except openai.AuthenticationError as e:
        _log.error(f"OpenAI API authentication failed for '{image_original_filename}': {e}", exc_info=True)
        raise HTTPException(status_code=401, detail=f"OpenAI API authentication failed. Check API key configuration: {e}")
    except openai.APIStatusError as e:
        _log.error(f"OpenAI API returned an error status {e.status_code} for '{image_original_filename}': {e.response}", exc_info=True)
        raise HTTPException(status_code=e.status_code or 500, detail=f"OpenAI API error: {e.message or e.response.text}")
    except HTTPException: # Re-raise HTTPExceptions from encode_image_to_base64
        raise
    except Exception as e:
        _log.error(f"An unexpected error occurred with the OpenAI API call for '{image_original_filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred with the OpenAI API: {type(e).__name__}")


async def run_ocr_and_llm_pipeline_for_api(image_path: Path, image_original_filename: str) -> str:
    """
    Full pipeline for API: runs Docling OCR, then sends text and image to LLM.
    Accepts a Path to a temporary image file. Raises HTTPException on failure.
    """
    if not image_path.exists():
        _log.error(f"Cannot run OCR, input file not found at temporary path: {image_path}")
        raise HTTPException(status_code=500, detail="Internal server error: Temporary image file for OCR not found.")

    _log.info(f"Initializing Docling to process image file: {image_path} (original: {image_original_filename})")
    try:
        # TODO: Consider making OCR language configurable via API parameter
        ocr_options = TesseractCliOcrOptions(lang=["auto"], force_full_page_ocr=True)
        pipeline_options = PdfPipelineOptions(do_ocr=True, force_full_page_ocr=True, ocr_options=ocr_options)
        
        # Docling typically infers InputFormat from file extension or content.
        # Explicitly setting InputFormat.IMAGE as this endpoint is for images.
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

        _log.info(f"Starting OCR on image: {image_path}...")
        # Note: doc_converter.convert is a synchronous (blocking) call.
        # For high-concurrency, run in a thread pool:
        # from fastapi.concurrency import run_in_threadpool
        # conv_result = await run_in_threadpool(doc_converter.convert, image_path)
        conv_result = doc_converter.convert(image_path)

        if conv_result and conv_result.status == ConversionStatus.SUCCESS:
            _log.info(f"OCR successful for '{image_original_filename}'. Extracting raw text.")
            raw_md_content = conv_result.document.export_to_markdown()

            if not raw_md_content or raw_md_content.isspace():
                _log.warning(f"OCR process for '{image_original_filename}' resulted in empty text.")
                raise HTTPException(status_code=422, detail="OCR process resulted in empty text. Cannot proceed to LLM structuring.")

            final_content = await structure_with_multimodal_llm(raw_md_content, image_path, image_original_filename)
            # structure_with_multimodal_llm raises HTTPException on its own errors

            _log.info(f"✅ Final structured Markdown output generated for '{image_original_filename}'")
            return final_content
        else:
            status_name = conv_result.status.name if conv_result and conv_result.status else "UNKNOWN_OR_NO_RESULT"
            error_msg = conv_result.error_message if conv_result and conv_result.error_message else "No specific error message from Docling."
            _log.error(f"Docling OCR failed for '{image_original_filename}' with status: {status_name}. Details: {error_msg}")
            raise HTTPException(status_code=500, detail=f"OCR processing failed with status: {status_name}. {error_msg}")

    except HTTPException: # Re-raise HTTPExceptions from called functions
        raise
    except Exception as e:
        _log.error(f"An exception occurred during the API pipeline for '{image_original_filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during processing: {type(e).__name__}")

# --- Pydantic Models for Request/Response ---
class OCRResponse(BaseModel):
    filename: str
    content_type: str
    markdown_output: str

class HealthCheckResponse(BaseModel):
    status: str
    message: str


# --- API Routes ---
@app.post("/process-image/",
          response_model=OCRResponse,
          summary="Process an Image for OCR and Structuring",
          tags=["OCR Processing"])
async def create_upload_file_and_process(
    file: UploadFile = File(..., description="Image file to be processed (e.g., JPEG, PNG).")
):
    """
    Upload an image, perform OCR, and structure the content using GPT-4o.

    - **Supported image types**: Common formats like JPEG, PNG, GIF, WEBP, BMP.
    - The image is processed by **Tesseract OCR** via the Docling library.
    - The OCR'd text and the original image are sent to **OpenAI GPT-4o** for structuring into Markdown.
    - Ensure the `OPENAI_API_KEY` environment variable is set on the server.
    """
    _log.info(f"Received file upload: '{file.filename}', Content-Type: '{file.content_type}'")

    # Basic content type validation
    if not file.content_type or not file.content_type.startswith("image/"):
        _log.warning(f"Uploaded file '{file.filename}' is not a valid image. Content-Type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"File provided is not an image or content type is missing. Detected: {file.content_type}. Please upload a valid image (e.g., JPEG, PNG)."
        )

    tmp_file_path: Path | None = None
    try:
        # Save uploaded file to a temporary location for processing
        suffix = Path(file.filename).suffix
        if not suffix: # Ensure there's a suffix for MIME type detection later
            suffix = ".tmp" # A generic suffix if none; OpenAI might struggle without a proper hint.
                            # Consider adding a suffix based on detected file.content_type if file.filename lacks one.

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="ocr_upload_") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = Path(tmp_file.name)
        _log.info(f"Uploaded file '{file.filename}' saved temporarily to '{tmp_file_path}'")

        # Run the full OCR and LLM pipeline
        markdown_result = await run_ocr_and_llm_pipeline_for_api(tmp_file_path, file.filename)
        
        return OCRResponse(
            filename=file.filename,
            content_type=file.content_type,
            markdown_output=markdown_result
        )
    except HTTPException as e:
        # Log HTTPExceptions specifically if needed, then re-raise
        _log.error(f"HTTPException during processing '{file.filename}': {e.status_code} - {e.detail}", exc_info=False) # No need for full stack trace here
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        _log.error(f"Unexpected critical error during processing '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {type(e).__name__}. Please contact support if the issue persists.")
    finally:
        # Ensure uploaded file object is closed
        if file and hasattr(file, 'file') and not file.file.closed:
            file.file.close()
        # Clean up the temporary file
        if tmp_file_path and tmp_file_path.exists():
            try:
                os.unlink(tmp_file_path)
                _log.info(f"Temporary file '{tmp_file_path}' deleted.")
            except Exception as e_unlink:
                _log.error(f"Error deleting temporary file '{tmp_file_path}': {e_unlink}", exc_info=True)


@app.get("/",
         response_model=HealthCheckResponse,
         summary="API Root / Health Check",
         tags=["General"])
async def root():
    """Provides a basic health check / welcome message."""
    return HealthCheckResponse(status="ok", message="Welcome to the OCR and Document Structuring API. Visit /docs for API documentation.")

# --- Main block for running with Uvicorn ---
if __name__ == "__main__":
    # This is for development. For production, use a proper ASGI server like Gunicorn with Uvicorn workers.
    # Example: uvicorn main:app --reload --host 0.0.0.0 --port 8000
    
    _log.info("Starting OCR and Document Structuring API...")
    if not os.getenv("OPENAI_API_KEY"):
        _log.warning("CRITICAL: OPENAI_API_KEY environment variable is not set. The LLM structuring will fail.")
        # Consider exiting if this key is absolutely critical for all operations:
        # import sys
        # sys.exit("Error: OPENAI_API_KEY is not set.")
    
    # You can configure host, port, and other uvicorn settings here or via command line
    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
