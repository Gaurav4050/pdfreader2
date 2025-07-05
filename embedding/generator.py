import os
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import GPT2TokenizerFast  # or any tokenizer for chunking
from transformers import AutoTokenizer

import sys
print("Python Executable:", sys.executable)


load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name="models/embedding-001")

        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def chunk_text_by_tokens(self, text, chunk_size=512, overlap=100):
        tokens = self.tokenizer.encode(text)
        chunks = []

        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks


    def generate_embeddings(self, chunks):
        embeddings = []
        for chunk in chunks:
            try:
                response = genai.embed_content(
                    model="models/embedding-001",
                    content=chunk,
                    task_type="retrieval_document"
                )
                embedding_vector = response['embedding']
                print(len(embedding_vector))
                embeddings.append(response['embedding'])
            except Exception as e:
                print(f"Embedding failed for a chunk: {e}")
        return embeddings
