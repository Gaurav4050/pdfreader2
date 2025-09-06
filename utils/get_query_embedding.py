from sentence_transformers import SentenceTransformer

# Load the same embedding model used during ingestion
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_query_embedding(query: str) -> list:
    """
    Generate an embedding for a given query using the same model as ingestion.
    
    Args:
        query (str): The query string to be embedded.
        
    Returns:
        list: The embedding vector for the query (dimension 384).
    """
    # Encode the query (returns numpy array)
    embedding = model.encode([query], convert_to_numpy=True)[0]
    return embedding.tolist()  # Convert to list for Milvus
