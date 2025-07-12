import os
import requests
from bs4 import BeautifulSoup
from utils.hash_utils import sha256_checksum, is_already_processed, mark_as_processed
from utils.pdf_utils import extract_text_from_pdf
from embedding.generator import EmbeddingGenerator
from vectorstore.milvus_client import MilvusClient
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
import uuid
import re
from selenium.webdriver.common.by import By

TEMP_DIR = "temp_pdfs"
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_file_ids_from_folder(folder_url):
    from selenium.webdriver.common.by import By

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # Required for newer headless Chrome
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(folder_url)
    time.sleep(5)  # Wait for JS-rendered content

    file_ids = set()

    # Files are rendered as elements with data-id attributes
    try:
        items = driver.find_elements(By.CSS_SELECTOR, 'div[data-id]')
        for item in items:
            file_id = item.get_attribute("data-id")
            if file_id:
                file_ids.add(file_id)
    except Exception as e:
        print("Error while extracting file ids:", e)

    driver.quit()
    return list(file_ids)

def download_pdf_by_id(file_id, dest_folder="downloads"):
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        print(f"Downloading file ID: {file_id} from {url}")
        response = requests.get(url)
        os.makedirs(dest_folder, exist_ok=True)
        file_path = os.path.join(dest_folder, f"{file_id}.pdf")
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    except Exception as e:
        print(f"Error downloading file ID {file_id}: {e}")
        return None

def extract_drive_file_id(url: str) -> str | None:
    # Extract file ID using regex
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None


def download_pdf_from_url(url: str, save_dir: str) -> str | None:
    try:
        file_id = extract_drive_file_id(url)
        if not file_id:
            print("❌ Invalid Google Drive link format.")
            return None

        # Construct direct download link
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        response = requests.get(download_url)
        if response.status_code != 200:
            print(f"❌ Failed to fetch file from Google Drive: HTTP {response.status_code}")
            return None

        # Use file_id as safe filename
        filename = os.path.join(save_dir, f"{file_id}.pdf")

        with open(filename, "wb") as f:
            f.write(response.content)

        return filename
    except Exception as e:
        print(f"Download error: {e}")
        return None


def extract_file_ids_and_names(folder_url: str):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(folder_url)
    time.sleep(5)  # Allow page to load

    results = []

    try:
        items = driver.find_elements(By.CSS_SELECTOR, 'div[data-id]')
        for item in items:
            file_id = item.get_attribute("data-id")
            file_name = item.get_attribute("aria-label") or item.text.strip()
            if file_id and file_name:
                results.append({"file_id": file_id, "file_name": file_name})
    except Exception as e:
        print("Error while extracting file details:", e)

    driver.quit()
    return results


def ingest_from_drive_folder(folder_url: str):
    print(f"Starting ingestion from folder: {folder_url}")
    file_ids = extract_file_ids_from_folder(folder_url)
    print(f"Found {len(file_ids)} files in the folder.")
    for file_id in file_ids:
        print(f"Processing file ID: {file_id}")
    embedder = EmbeddingGenerator()
    milvus = MilvusClient()

    # download_pdf_by_id
    for file_id in file_ids:
        path = download_pdf_by_id(file_id, TEMP_DIR)
        if not path:
            print(f"❌ Failed to download file ID: {file_id}")
            continue
        # checksum = sha256_checksum(path)
        # if is_already_processed(checksum):
        #     print(f"❌ Skipping duplicate: {file_id}")
        #     continue
        print(f"✅ Processing: {file_id}")
        text = extract_text_from_pdf(path)
        print(f"Extracted text from {file_id} with length {len(text)}")
        chunks = embedder.chunk_text_by_tokens(text)
        embeddings = embedder.generate_embeddings(chunks)

        
        pdf_ids = []
        file_ids = []
        chunks_list = []
        embeddings_list = []

        for chunk, embedding in zip(chunks, embeddings):
            pdf_id = str(uuid.uuid4())  # Generate a unique PDF ID
            pdf_ids.append(pdf_id)
            file_ids.append(file_id)
            chunks_list.append(chunk)
            embeddings_list.append(embedding)

        milvus.insert({"pdf_id": pdf_ids,"file_id": file_ids,"chunk": chunks_list,"embedding": embeddings_list})


        # mark_as_processed(checksum)

    # for file_id in file_ids:
    #     path = download_pdf_from_drive(file_id)
    #     if not path:
    #         continue
    #     checksum = sha256_checksum(path)
    #     if is_already_processed(checksum):
    #         print(f"❌ Skipping duplicate: {file_id}")
    #         continue
    #     print(f"✅ Processing: {file_id}")
    #     text = extract_text_from_pdf(path)
    #     chunks = embedder.chunk_text_by_tokens(text)
    #     embeddings = embedder.generate_embeddings(chunks)

    #     for chunk, embedding in zip(chunks, embeddings):
    #         milvus.insert({
    #             "pdf_id": file_id,
    #             "chunk": chunk,
    #             "embedding": embedding
    #         })
    #     mark_as_processed(checksum)



def ingest_single_public_pdf(pdf_url: str):
    print(f"📥 Starting ingestion for PDF URL: {pdf_url}")

    path = download_pdf_from_url(pdf_url, TEMP_DIR)
    if not path:
        print(f"❌ Failed to download PDF from URL: {pdf_url}")
        return

    print(f"✅ Downloaded PDF to: {path}")

    embedder = EmbeddingGenerator()
    milvus = MilvusClient()

    text = extract_text_from_pdf(path)
    if not text.strip():
        print("⚠️ Empty or unreadable text. Skipping.")
        return

    print(f"📚 Extracted {len(text)} characters from PDF")

    chunks = embedder.chunk_text_by_tokens(text)  # enforce limit
    embeddings = embedder.generate_embeddings(chunks)

    if len(chunks) != len(embeddings):
        print("❌ Mismatch between chunks and embeddings.")
        return

    pdf_id = str(uuid.uuid4())
    file_id = os.path.basename(path)  # use filename as file_id

    milvus.insert({
        "pdf_id": [pdf_id] * len(chunks),
        "file_id": [file_id] * len(chunks),  # ✅ THIS LINE FIXES THE ERROR
        "chunk": chunks,
        "embedding": embeddings,
    })

    print(f"✅ Ingested {len(chunks)} chunks for file: {file_id}")

