import os
import re
import io
import uuid
import logging
import fitz  # PyMuPDF
import unicodedata
from PIL import Image
import pytesseract
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from utils.hash_utils import sha256_checksum, is_already_processed, mark_as_processed
from vectorstore.milvus_client import MilvusClient
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from utils.llm_client import TextGenerator
import base64
import numpy as np
import google.generativeai as genai


# -------------------------------------------------
# Logging setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_DIR = "temp_pdfs"
os.makedirs(TEMP_DIR, exist_ok=True)


# -------------------------------------------------
# PDF Processor
# -------------------------------------------------
class EnhancedPDFProcessor:
    def __init__(self):
        self.languages = "eng+hin+urd+ben+guj+pan+tam+tel+kan+mal+ori+asm"
        self.tesseract_config = "--oem 3 --psm 6"

        # 👉 NEW: Hindi normalizer instance
        factory = IndicNormalizerFactory()
        self.hindi_normalizer = factory.get_normalizer("hi")

        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        self.text_generator = TextGenerator(api_key)

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text: remove [UNK], �, normalize Unicode, Indic normalization.
        Works for both Hindi and English.
        """
        if not text:
            return ""

        # Remove tokenizer junk
        text = text.replace("[UNK]", "")

        # Normalize Unicode (NFC form)
        text = unicodedata.normalize("NFC", text)

        # Remove replacement characters
        text = text.replace("�", "")

        # Remove weird control characters
        text = re.sub(r"[\x00-\x1F\x7F]", " ", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        # 👉 Apply Indic normalization (mainly helps for Hindi/Indic scripts)
        try:
            text = self.hindi_normalizer.normalize(text)
        except Exception as e:
            logger.warning(f"Indic normalization failed: {e}")

        return text

    def is_text_readable(self, text, min_readable_ratio=0.3):
        if not text or len(text.strip()) < 10:
            return False
        meaningful_chars = sum(
            c.isalnum() or c.isspace() or "\u0900" <= c <= "\u0D7F" for c in text
        )
        ratio = meaningful_chars / len(text)
        return ratio >= min_readable_ratio

    def extract_text_with_pymupdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return ""

    def enhance_image_for_ocr(self, image):
        try:
            if image.mode != "L":
                image = image.convert("L")
            from PIL import ImageEnhance, ImageFilter

            image = ImageEnhance.Contrast(image).enhance(2.0)
            image = image.filter(ImageFilter.SHARPEN)

            w, h = image.size
            if w < 1000:
                scale = 1000 / w
                image = image.resize(
                    (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
                )
            return image
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image

    def extract_text_with_ocr(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                img = self.enhance_image_for_ocr(img)
                page_text = pytesseract.image_to_string(
                    img, lang=self.languages, config=self.tesseract_config
                )
                text += page_text + "\n"
            doc.close()
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def extract_text_smart(self, pdf_path):
        logger.info(f"Extracting text from {pdf_path}")
        text = self.extract_text_with_pymupdf(pdf_path)
        if self.is_text_readable(text):
            return text
        return self.extract_text_with_ocr(pdf_path)


# -------------------------------------------------
# Embedding Generator
# -------------------------------------------------
class EmbeddingGenerator:
    def __init__(self):
        # Initialize Gemini API for embeddings
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        genai.configure(api_key=self.api_key)

    def chunk_text_by_tokens(self, text, max_tokens=256):
        words, chunks, current = text.split(), [], []
        for w in words:
            if len(current) + 1 > max_tokens:
                chunks.append(" ".join(current))
                current = []
            current.append(w)
        if current:
            chunks.append(" ".join(current))
        return chunks

    def generate_embeddings(self, chunks):
        # Generate 768-dimensional embeddings using Gemini embedding-001 model
        embeddings = []
        for chunk in chunks:
            try:
                response = genai.embed_content(
                    model="models/embedding-001",
                    content=chunk,
                    task_type="retrieval_query",
                    output_dimensionality=768
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                logger.error(f"Gemini embedding generation failed: {e}")
                # Return a zero vector as fallback
                embeddings.append(np.zeros(768))
        return np.array(embeddings)

    def generate_gemini_embeddings(self, chunks):  # Keeping for backward compatibility
        # Generate 768-dimensional embeddings using Gemini embedding-001 model
        embeddings = []
        for chunk in chunks:
            try:
                response = genai.embed_content(
                    model="models/embedding-001",
                    content=chunk,
                    task_type="retrieval_query",
                    output_dimensionality=768
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                logger.error(f"Gemini embedding generation failed: {e}")
                # Return a zero vector as fallback
                embeddings.append(np.zeros(768))
        return np.array(embeddings)


# -------------------------------------------------
# Google Drive API
# -------------------------------------------------
class GoogleDriveAPIClient:
    def __init__(self, service_account_path=None, api_key=None):
        if service_account_path and os.path.exists(service_account_path):
            creds = Credentials.from_service_account_file(
                service_account_path,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )
            self.service = build("drive", "v3", credentials=creds)
        elif api_key:
            self.service = build("drive", "v3", developerKey=api_key)
        else:
            raise ValueError("Need service_account_path or api_key")

    def extract_folder_id(self, folder_url):
        patterns = [
            r"/folders/([a-zA-Z0-9_-]+)",
            r"id=([a-zA-Z0-9_-]+)",
            r"/drive/folders/([a-zA-Z0-9_-]+)",
        ]
        for p in patterns:
            m = re.search(p, folder_url)
            if m:
                return m.group(1)
        return None

    def list_files_in_folder(self, folder_id, mime_type="application/pdf"):
        try:
            q = f"'{folder_id}' in parents and mimeType='{mime_type}'"
            res = self.service.files().list(q=q, fields="files(id,name)").execute()
            return res.get("files", [])
        except Exception as e:
            logger.error(f"List files failed: {e}")
            return []

    def download_file(self, file_id, dest_path):
        try:
            req = self.service.files().get_media(fileId=file_id)
            with open(dest_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, req)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            return dest_path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None


def get_service_account_path():
    key_b64 = os.environ.get("GCP_KEY")
    key_path = "/tmp/gcp-key.json"  # Render पर /tmp writable होता है
    if key_b64:
        with open(key_path, "wb") as f:
            f.write(base64.b64decode(key_b64))
    return key_path
# -------------------------------------------------
# Ingestion
# -------------------------------------------------
def ingest_from_drive_folder_enhanced(folder_url, service_account_path=None, api_key=None):
    logger.info(f"🚀 Starting ingestion from: {folder_url}")
    # service_account_path = get_service_account_path()

    # for local testing
    service_account_path="./pdfreadergenai-dd6b7e9bb5ab.json"
    try:
        drive = GoogleDriveAPIClient(service_account_path, api_key)
        processor = EnhancedPDFProcessor()
        embedder = EmbeddingGenerator()
        milvus = MilvusClient()
    except Exception as e:
        logger.error(f"Init failed: {e}")
        return

    folder_id = drive.extract_folder_id(folder_url)
    if not folder_id:
        logger.error("Invalid folder URL")
        return

    files = drive.list_files_in_folder(folder_id)
    if not files:
        logger.warning("No PDF files found")
        return

    for i, f in enumerate(files, 1):
        file_id, file_name = f["id"], f["name"]
        logger.info(f"[{i}/{len(files)}] Processing {file_name}")

        path = os.path.join(TEMP_DIR, f"{file_id}.pdf")
        if not drive.download_file(file_id, path):
            continue

        try:
            text = processor.extract_text_smart(path)
            if not text.strip():
                continue

            chunks = embedder.chunk_text_by_tokens(text)
            embeddings = embedder.generate_embeddings(chunks)

            milvus.insert(
                {
                    "pdf_id": [str(uuid.uuid4()) for _ in chunks],
                    "file_id": [file_id] * len(chunks),
                    "chunk": chunks,
                    "embedding": embeddings,
                }
            )
            logger.info(f"✅ Ingested {len(chunks)} chunks for {file_name}")

        except Exception as e:
            logger.error(f"Error {file_name}: {e}")
        finally:
            if os.path.exists(path):
                os.remove(path)

    logger.info("🎉 Ingestion completed!")


# -------------------------------------------------
# Usage
# -------------------------------------------------
if __name__ == "__main__":
    ingest_from_drive_folder_enhanced(
        folder_url="https://drive.google.com/drive/folders/1tm8shLBaCtWBFhFoVEngsMxUSOHtCZ64?usp=sharing",
        service_account_path="./pdfreadergenai-dd6b7e9bb5ab.json",
    )
