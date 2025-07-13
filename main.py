from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ingest.drive_folder_ingest import ingest_from_drive_folder, extract_file_ids_and_names, ingest_single_public_pdf
from utils.get_query_embedding import get_query_embedding
from utils.serach_chunks import search_chunks
from utils.group_by_file_id import group_by_file_id
from utils.summarize_results_with_model import summarize_results_with_model
from utils.summarize_keyword_results_with_model import summarize_keyword_results_with_model
import os
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
from utils.rerank_results_with_model import rerank_results_with_model_parallel
import requests
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict
import json
app = FastAPI()
from fastapi.responses import JSONResponse
from pathlib import Path
import uvicorn

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ✅ Allow all origins (use only in development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    folder_url: str

class IngestRequestSingle(BaseModel):
    pdf_url: str

class QueryRequest(BaseModel):
    query: str
    mode: Literal["conceptual", "keyword"] = "conceptual"


@app.post("/ingest-drive-folder")
def ingest_drive_folder(req: IngestRequest):
    ingest_from_drive_folder(req.folder_url)
    return {"status": "success", "message": "Folder ingested"}

@app.post("/ingest-single-public-pdf")
def ingest_single_pdf(req: IngestRequestSingle):
    ingest_single_public_pdf(req.pdf_url)
    return {"status": "success", "message": "Single public PDF ingested"}



@app.post("/semantic-search")
def semantic_search(req: QueryRequest):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in environment variables.")

        # 🔍 Embed query (same for both modes now)
        embedding = get_query_embedding(req.query)
        if not embedding:
            raise HTTPException(status_code=400, detail="Failed to generate embedding.")

        # 🔍 Search in Milvus
        raw_results = search_chunks(embedding)
        if not raw_results:
            raise HTTPException(status_code=404, detail="No results found.")

        #rerank_results_with_gemini
        try:
            reranked_results = rerank_results_with_model_parallel(req.query, raw_results, api_key, top_k=10)
        except Exception as e:
            print(f"Error during reranking: {e}")
            raise HTTPException(status_code=500, detail="Failed to rerank results with Gemini.")

        # 📚 Group top results by file_id
        top_results = group_by_file_id(reranked_results)
        # 🧠 Summarize differently based on mode
        if req.mode == "conceptual":
            summary = summarize_results_with_model(req.query, top_results, api_key)
        elif req.mode == "keyword":
            summary = summarize_keyword_results_with_model(req.query, raw_results, api_key)
        else:
            raise HTTPException(status_code=400, detail="Invalid search mode. Use 'conceptual' or 'keyword'.")

        return {
            "query": req.query,
            "summary": summary,
            "results": top_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation-results")
async def get_evaluation_results():
    file_path = Path("results.json")
    if not file_path.exists():
        return {"error": "result.json not found"}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Load ground truth at startup
with open("ground_truth.json", "r", encoding="utf-8") as f:
    GROUND_TRUTH = {item["query"]: item["relevant_file_ids"] for item in json.load(f)}

API_URL = "http://127.0.0.1:8000/semantic-search"  # Update if hosted remotely

@app.get("/evaluate-ground-truth")
def evaluate_ground_truth():
    metrics = {
        "conceptual": {"precision": [], "recall": [], "f1": []},
        "keyword": {"precision": [], "recall": [], "f1": []}
    }

    detailed_results = []

    for query, relevant_ids in GROUND_TRUTH.items():
        result = {"query": query, "modes": {}}

        for mode in ["conceptual", "keyword"]:
            try:
                response = requests.post(API_URL, json={"query": query, "mode": mode})
                response.raise_for_status()
                data = response.json()
                predicted_ids = [r["file_id"] for r in data["results"]]

                all_ids = list(set(relevant_ids + predicted_ids))
                y_true = [1 if fid in relevant_ids else 0 for fid in all_ids]
                y_pred = [1 if fid in predicted_ids else 0 for fid in all_ids]

                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                metrics[mode]["precision"].append(precision)
                metrics[mode]["recall"].append(recall)
                metrics[mode]["f1"].append(f1)

                result["modes"][mode] = {
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "f1_score": round(f1, 3),
                    "predicted_file_ids": predicted_ids
                }
            except Exception as e:
                result["modes"][mode] = {"error": str(e)}

        detailed_results.append(result)

    summary = {}
    for mode in ["conceptual", "keyword"]:
        try:
            p_avg = sum(metrics[mode]["precision"]) / len(metrics[mode]["precision"])
            r_avg = sum(metrics[mode]["recall"]) / len(metrics[mode]["recall"])
            f_avg = sum(metrics[mode]["f1"]) / len(metrics[mode]["f1"])
        except ZeroDivisionError:
            p_avg = r_avg = f_avg = 0.0

        summary[mode] = {
            "avg_precision": round(p_avg, 3),
            "avg_recall": round(r_avg, 3),
            "avg_f1_score": round(f_avg, 3),
        }

    return {
        "summary": summary,
        "details": detailed_results
    }



@app.get("/list-drive-files/")
def list_drive_files(folder_url: str):
    """
    Extract and return list of files (file_id + file_name) from a Google Drive folder.
    """
    try:
        files = extract_file_ids_and_names(folder_url)
        return JSONResponse(content={"count": len(files), "files": files})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Define request format
class EvalItem(BaseModel):
    query: str

class EvalBatchRequest(BaseModel):
    queries: List[EvalItem]


