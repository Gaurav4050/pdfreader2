import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment.")

def get_query_embedding(query: str, output_dim: int = 768) -> list:
    """
    Generate an embedding for a given query using Google Gemini API with a specified output dimension.
    
    Args:
        query (str): The query string to be embedded.
        output_dim (int): The desired output dimension of the embedding (768, 1536, or 3072).
        
    Returns:
        list: The embedding vector for the query.
    """
    genai.configure(api_key=api_key)

    response = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query",
        output_dimensionality=output_dim  # Specify the output dimension
    )
    return response['embedding']
