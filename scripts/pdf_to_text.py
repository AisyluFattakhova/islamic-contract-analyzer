import os
import sys
from pathlib import Path
import pdfplumber
import pandas as pd
from tqdm import tqdm

# Import text processing utilities
from text_utils import clean_text

# Get project root directory (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_DIR = PROJECT_ROOT / "raw_pdf"
OUT_DIR = PROJECT_ROOT / "clean_text"
META_OUT = PROJECT_ROOT / "metadata" / "documents.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
META_OUT.parent.mkdir(parents=True, exist_ok=True)

def extract_pdf(file_path: Path):
    text_pages = []
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"  Extracting text from {total_pages} pages...")
            for i, page in enumerate(tqdm(pdf.pages, desc="  Pages", leave=False, total=total_pages), 1):
                try:
                    page_text = page.extract_text() or ""
                    text_pages.append(page_text)
                    if i % 50 == 0:  # Print progress every 50 pages
                        print(f"  Processed {i}/{total_pages} pages...")
                except Exception as e:
                    print(f"  [WARNING] Error extracting page {i}: {e}")
                    text_pages.append("")  # Add empty string for failed page
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, 0
    full_text = "\n\n".join(text_pages)
    num_pages = len(text_pages)
    print(f"  Extracted {len(full_text)} characters from {num_pages} pages")
    return full_text, num_pages

def main():
    # Debug: Show paths being used
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Looking for PDFs in: {RAW_DIR}")
    print(f"Output directory: {OUT_DIR}")
    
    # Check if directory exists
    if not RAW_DIR.exists():
        print(f"[ERROR] Directory does not exist: {RAW_DIR}")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    # Get list of PDF files
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    if not pdf_files:
        print(f"[WARNING] No PDF files found in {RAW_DIR}")
        return
    
    docs = []
    for pdf_file in tqdm(sorted(pdf_files), desc="Processing PDFs"):
        print(f"Processing: {pdf_file.name}")
        raw, num_pages = extract_pdf(pdf_file)
        if not raw:
            print(f"Skipping {pdf_file.name}, extraction failed.")
            continue
        cleaned = clean_text(raw, verbose=True)
        out_file = OUT_DIR / (pdf_file.stem + ".txt")
        print(f"  Writing output to: {out_file}")
        out_file.write_text(cleaned, encoding="utf-8")
        docs.append({
            "filename": pdf_file.name,
            "stem": pdf_file.stem,
            "path": str(out_file),
            "pages": num_pages
        })
    if docs:
        df = pd.DataFrame(docs)
        df.to_csv(META_OUT, index=False)
        print("Saved metadata to", META_OUT)
    else:
        print("No documents processed.")

if __name__ == "__main__":
    main()
