import os, json, re, io
from dotenv import load_dotenv
import pdfplumber
import camelot
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from tools.column_utils import group_words_into_columns, lines_from_words

load_dotenv()
TESS = os.getenv("TESSERACT_PATH")
if TESS and os.path.exists(TESS):
    pytesseract.pytesseract.tesseract_cmd = TESS

PDF_DIR = "./docs"
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)
CHUNKS_PATH = os.path.join(OUT_DIR, "chunks.jsonl")

FIGURE_REGEX = re.compile(r"^\s*(Figure|Fig\.|FIGURE)\b", re.IGNORECASE)

def save_chunk(writer, doc_id, kind, text, page_num, extra=None):
    if not text or not text.strip():
        return
    rec = {
        "doc_id": doc_id,
        "page": page_num,
        "kind": kind,              # "text" | "table" | "figure" | "figure_ocr"
        "text": text.strip()
    }
    if extra:
        rec.update(extra)
    writer.write(json.dumps(rec, ensure_ascii=False) + "\n")

def extract_text_two_column(page):
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    if not words:
        return []
    left, right = group_words_into_columns(words, page.width)
    left_lines = lines_from_words(left)
    right_lines = lines_from_words(right)
    # reading order: left column top→bottom, then right top→bottom
    return left_lines + right_lines

def extract_tables_camelot(pdf_path, page_number):
    # Camelot uses 1-based page numbers
    p = str(page_number)
    tables = []
    try:
        t = camelot.read_pdf(pdf_path, pages=p, flavor="lattice")
        tables.extend(t)
    except Exception:
        pass
    try:
        t = camelot.read_pdf(pdf_path, pages=p, flavor="stream")
        tables.extend(t)
    except Exception:
        pass
    texts = []
    for tb in tables:
        if tb.shape[0] == 0 or tb.shape[1] == 0:
            continue
        df = tb.df
        # stringify a compact representation
        rows = [" | ".join(map(str, df.iloc[i].tolist())) for i in range(len(df))]
        texts.append("\n".join(rows))
    return texts

def extract_figure_captions(lines):
    return [ln for ln in lines if FIGURE_REGEX.match(ln)]

def ocr_page_regions_for_figures(pdf_path, page_idx, dpi=200):
    """
    As a simple heuristic, OCR the bottom quarter of the page where captions often live,
    plus the central band (where many schematics have labels).
    """
    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_idx+1, last_page=page_idx+1)
    if not images:
        return []
    page_img = images[0]
    w, h = page_img.size
    regions = [
        (0, int(0.70*h), w, h),                # bottom ~30% (captions)
        (int(0.15*w), int(0.30*h), int(0.85*w), int(0.70*h)),  # central band
    ]
    texts = []
    for (x0,y0,x1,y1) in regions:
        crop = page_img.crop((x0,y0,x1,y1))
        txt = pytesseract.image_to_string(crop)
        if txt and txt.strip():
            texts.append(txt.strip())
    return texts

def main():
    with open(CHUNKS_PATH, "w", encoding="utf-8") as writer:
        for name in sorted(os.listdir(PDF_DIR)):
            if not name.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(PDF_DIR, name)
            doc_id = name
            print(f"Parsing {doc_id} ...")
            with pdfplumber.open(pdf_path) as pdf:
                for p_idx, page in enumerate(pdf.pages):
                    # TEXT
                    lines = extract_text_two_column(page)
                    if lines:
                        save_chunk(writer, doc_id, "text", "\n".join(lines), p_idx+1)

                    # FIGURE captions (from text lines)
                    captions = extract_figure_captions(lines)
                    for cap in captions:
                        save_chunk(writer, doc_id, "figure", cap, p_idx+1)

                    # TABLES
                    tables_txt = extract_tables_camelot(pdf_path, p_idx+1)
                    for ttxt in tables_txt:
                        save_chunk(writer, doc_id, "table", ttxt, p_idx+1)

                    # FIGURE OCR (labels inside diagrams)
                    ocr_texts = ocr_page_regions_for_figures(pdf_path, p_idx)
                    for ot in ocr_texts:
                        save_chunk(writer, doc_id, "figure_ocr", ot, p_idx+1)

    print(f"Wrote chunks → {CHUNKS_PATH}")

if __name__ == "__main__":
    main()