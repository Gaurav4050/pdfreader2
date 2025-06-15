import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment.")

def get_query_embedding(query: str) -> list:

    """
    Generate an embedding for a given query using Google Gemini API.
    Args:
        query (str): The query string to be embedded.
    Returns:
        list: The embedding vector for the query.
    """
    
    genai.configure(api_key=api_key)



    response = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return response['embedding']
