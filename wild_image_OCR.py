import os
import openai
from pathlib import Path
import logging
import time
import base64

# Docling imports
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# --- Configuration ---
INPUT_IMAGE_PATH = Path("test_1.jpeg") # A bill of lading image works well here
OUTPUT_DIR = Path("final_multimodal_output")
FINAL_MARKDOWN_PATH = OUTPUT_DIR / "structured_document.md"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
_log = logging.getLogger(__name__)


def encode_image_to_base64(image_path: Path) -> str:
    """Reads an image file and encodes it as a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def structure_with_multimodal_llm(raw_text: str, image_path: Path) -> str | None:
    """
    Sends raw OCR text AND the original image to GPT-4o to be cleaned,
    structured, and formatted, using the image as a visual reference.
    """
    _log.info("Sending OCR text and source image to OpenAI GPT-4o for structuring...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _log.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
        
    client = openai.OpenAI(api_key=api_key)

    # Encode the local image to Base64
    try:
        base64_image = encode_image_to_base64(image_path)
    except FileNotFoundError:
        _log.error(f"Image file not found at {image_path} for LLM processing.")
        return None

    # This new prompt explicitly tells the LLM to use the image as the ground truth.
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
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                # Pass the Base64 encoded image data directly
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1, # Very low temperature to prevent creativity/hallucination
            max_tokens=2000
        )
        end_time = time.time() - start_time
        _log.info(f"Multimodal LLM processing complete in {end_time:.2f} seconds.")
        
        structured_content = response.choices[0].message.content
        return structured_content

    except Exception as e:
        _log.error(f"An error occurred with the OpenAI API call: {e}", exc_info=True)
        return None


def run_ocr_and_llm_pipeline(image_path: Path, output_md_path: Path):
    """
    Full pipeline: runs Docling OCR, then sends the text and source image to an LLM.
    """
    if not image_path.exists():
        _log.error(f"Cannot run OCR, input file not found: {image_path}")
        return

    _log.info("Initializing Docling to process image file...")
    try:
        ocr_options = TesseractCliOcrOptions(lang=["auto"], force_full_page_ocr=True)
        pipeline_options = PdfPipelineOptions(do_ocr=True, force_full_page_ocr=True, ocr_options=ocr_options)
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

        _log.info(f"Starting OCR on image: {image_path}...")
        conv_result = doc_converter.convert(image_path)

        if conv_result and conv_result.status == ConversionStatus.SUCCESS:
            _log.info("OCR successful. Extracting raw text.")
            raw_md_content = conv_result.document.export_to_markdown()

            if not raw_md_content or raw_md_content.isspace():
                _log.warning("OCR process resulted in empty text. Cannot send to LLM.")
                return

            # --- UPDATED STEP: Pass both text AND the image path to the LLM function ---
            final_content = structure_with_multimodal_llm(raw_md_content, image_path)

            if final_content:
                output_md_path.write_text(final_content, encoding="utf-8")
                _log.info(f"✅ Final structured Markdown output saved to: {output_md_path}")
            else:
                _log.error("LLM structuring failed. Final file not saved.")

        else:
            status = conv_result.status.name if conv_result else "UNKNOWN"
            _log.error(f"Docling OCR failed with status: {status}.")

    except Exception as e:
        _log.error(f"An exception occurred during the pipeline: {e}", exc_info=True)


def main():
    """Main pipeline execution."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_IMAGE_PATH.exists():
        _log.error(f"Input image '{INPUT_IMAGE_PATH}' not found!")
        return

    run_ocr_and_llm_pipeline(INPUT_IMAGE_PATH, FINAL_MARKDOWN_PATH)


if __name__ == "__main__":
    main()