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
