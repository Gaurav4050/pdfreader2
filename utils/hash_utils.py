import hashlib
import os
import json

HASH_DB = "processed_pdfs.json"

def sha256_checksum(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def is_already_processed(checksum):
    if not os.path.exists(HASH_DB):
        return False
    with open(HASH_DB, "r") as f:
        return checksum in json.load()

def mark_as_processed(checksum):
    if os.path.exists(HASH_DB):
        with open(HASH_DB, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(checksum)
    with open(HASH_DB, "w") as f:
        json.dump(data, f)
