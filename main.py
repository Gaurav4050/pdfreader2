from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ingest.drive_folder_ingest import ingest_from_drive_folder
from utils.get_query_embedding import get_query_embedding
from utils.serach_chunks import search_chunks
from utils.group_by_file_id import group_by_file_id
from utils.summarize_results_with_gemini import summarize_results_with_gemini
import os


app = FastAPI()

class IngestRequest(BaseModel):
    folder_url: str

class QueryRequest(BaseModel):
    query: str

@app.post("/ingest-drive-folder")
def ingest_drive_folder(req: IngestRequest):
    ingest_from_drive_folder(req.folder_url)
    return {"status": "success", "message": "Folder ingested"}


@app.post("/semantic-search")
def semantic_search(req: QueryRequest):
    try:
        embedding = get_query_embedding(req.query)
        if not embedding:
            raise HTTPException(status_code=400, detail="Failed to generate embedding for the query.")
        
        raw_results = search_chunks(embedding)
        if not raw_results:
            raise HTTPException(status_code=404, detail="No results found for the query.")
        
        top_results = group_by_file_id(raw_results)
        if not top_results:
            raise HTTPException(status_code=404, detail="No grouped results found.")

        # ✨ Generate summary using LLM
        summary = summarize_results_with_gemini(req.query, top_results, api_key=os.getenv("GEMINI_API_KEY"))

        return {
            "query": req.query,
            "summary": summary,
            "results": top_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

