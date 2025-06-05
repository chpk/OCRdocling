import json
import logging
import time
import re
from pathlib import Path
from urllib.parse import urlparse

# Docling imports
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# --- Configuration ---
INPUT_PDF_DIR = Path("/home/motormaven/Desktop/Doc_Parsing/notices_pdfs-20250530T200719Z-1-001/input_processing_PDFs/")
MAIN_OUTPUT_DIR = Path("output_data_folder")
MARKDOWN_OUTPUT_DIR = MAIN_OUTPUT_DIR / "markdown_files"
RESULTS_JSON_PATH = MAIN_OUTPUT_DIR / "processing_results.json"
UNPROCESSED_LOG_PATH = MAIN_OUTPUT_DIR / "unprocessed_files.txt"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
_log = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """Sanitizes a filename to be valid."""
    name = str(filename).strip().replace(" ", "_")
    name = re.sub(r"(?u)[^-\w.]", "", name)
    name = name.removesuffix(".pdf") # Remove extension for cleaner naming
    name = name[:100] # Limit length
    return name if name else "untitled"


def main():
    """
    Main function to process PDFs from local folder, perform OCR, and save results.
    """
    start_total_time = time.time()
    # Create output directories
    MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Initialize Docling Converter (once for efficiency) ---
    _log.info("Initializing DocumentConverter with robust OCR settings...")
    try:
        ocr_options = TesseractCliOcrOptions(lang=["eng", "lat", "hin", "tel"], force_full_page_ocr=True)
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            force_full_page_ocr=True,
            ocr_options=ocr_options,
            do_table_structure=True,
            table_structure_options={"do_cell_matching": True}
        )
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    except Exception as e:
        _log.error(f"Fatal error initializing Docling DocumentConverter: {e}")
        _log.error("Please ensure docling and its dependencies (like Tesseract) are installed correctly.")
        return

    # --- Process PDFs ---
    overall_results = {}
    unprocessed_files = []
    processed_count = 0
    failed_count = 0

    # Get all PDF files from the input directory
    pdf_files = list(INPUT_PDF_DIR.glob("**/*.pdf"))
    
    if not pdf_files:
        _log.error(f"No PDF files found in {INPUT_PDF_DIR}")
        return

    _log.info(f"Found {len(pdf_files)} PDF files to process")

    for pdf_path in pdf_files:
        _log.info(f"--- Starting processing for: {pdf_path} ---")
        
        try:
            # Use docling to convert the local PDF file
            start_time = time.time()
            conv_result = doc_converter.convert(str(pdf_path))
            end_time = time.time() - start_time

            # Check the result of the conversion
            if conv_result and conv_result.status == ConversionStatus.SUCCESS:
                _log.info(f"Successfully converted in {end_time:.2f} seconds.")
                
                # Get the extracted data as Markdown
                doc = conv_result.document
                md_content = doc.export_to_markdown()

                # Save the Markdown file
                base_filename = sanitize_filename(pdf_path.name)
                md_filepath = MARKDOWN_OUTPUT_DIR / f"{base_filename}.md"
                with open(md_filepath, "w", encoding="utf-8") as f:
                    f.write(md_content)

                # Record the success
                overall_results[str(pdf_path)] = str(md_filepath)
                processed_count += 1
            
            else:
                # Handle conversion failure
                status = conv_result.status.name if conv_result else "UNKNOWN"
                _log.error(f"Failed to convert {pdf_path}. Status: {status}")
                unprocessed_files.append(f"{pdf_path} (Conversion Status: {status})")
                failed_count += 1

        except Exception as e:
            # Handle exceptions during the conversion process
            _log.error(f"An exception occurred while processing {pdf_path}: {e}")
            unprocessed_files.append(f"{pdf_path} (Exception: {e})")
            failed_count += 1
        
        _log.info(f"--- Finished processing: {pdf_path} ---")

    # --- Save Final Reports ---
    _log.info("Saving final processing reports...")
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=4)
    _log.info(f"✅ Success results saved to: {RESULTS_JSON_PATH}")

    if unprocessed_files:
        with open(UNPROCESSED_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("The following PDF files could not be processed:\n")
            for file_reason in unprocessed_files:
                f.write(f"- {file_reason}\n")
        _log.warning(f"⚠️ Some PDFs failed. Log of unprocessed files saved to: {UNPROCESSED_LOG_PATH}")
    else:
        _log.info("✅ All PDFs were processed successfully!")

    end_total_time = time.time() - start_total_time
    _log.info(f"\n--- 📊 Processing Summary ---")
    _log.info(f"Total time taken: {end_total_time:.2f} seconds.")
    _log.info(f"Successfully processed PDFs: {processed_count}")
    _log.info(f"Failed PDFs: {failed_count}")


if __name__ == "__main__":
    main()