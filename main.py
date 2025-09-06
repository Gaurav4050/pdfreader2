from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from ingest.drive_folder_ingest import ingest_from_drive_folder_enhanced
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
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import asyncio
import math

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
    ingest_from_drive_folder_enhanced(req.folder_url)
    return {"status": "success", "message": "Folder ingested"}

# @app.post("/ingest-single-public-pdf")
# def ingest_single_pdf(req: IngestRequestSingle):
#     ingest_single_public_pdf(req.pdf_url)
#     return {"status": "success", "message": "Single public PDF ingested"}



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




def get_predicted_file_ids(query: str) -> List[str]:
    print("Processing query:", query)

    embedding = get_query_embedding(query)
    if not embedding:
        print(f"Failed to generate embedding for query: {query}")
        raise HTTPException(status_code=400, detail="Failed to generate embedding.")

    raw_results = search_chunks(embedding)
    if not raw_results:
        print(f"No raw results found for query: {query}")
        return []

    api_key = os.getenv("GEMINI_API_KEY")
    try:
        reranked_results = rerank_results_with_model_parallel(query, raw_results, api_key, top_k=20)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {e}")

    top_results = group_by_file_id(reranked_results)
    file_ids = [r["file_id"] for r in top_results]
    return file_ids

# -----------------------
# Async wrapper for ThreadPoolExecutor
# -----------------------
executor = ThreadPoolExecutor(max_workers=10)  # adjust based on CPU/network

async def get_predicted_file_ids_async(query: str) -> List[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_predicted_file_ids, query)

# -----------------------
# Metrics calculation
# -----------------------
def precision_recall(predicted, expected):
    if not predicted:
        return 0, 0
    relevant_retrieved = sum([1 for p in predicted if p in expected])
    precision = relevant_retrieved / len(predicted)
    recall = relevant_retrieved / len(expected) if expected else 0
    return precision, recall

def average_precision(predicted, expected):
    ap = 0
    relevant_count = 0
    for i, p in enumerate(predicted):
        if p in expected:
            relevant_count += 1
            ap += relevant_count / (i + 1)
    return ap / len(expected) if expected else 0

def reciprocal_rank(predicted, expected):
    for i, p in enumerate(predicted):
        if p in expected:
            return 1 / (i + 1)
    return 0

def ndcg(predicted, expected):
    dcg = 0
    for i, p in enumerate(predicted):
        if p in expected:
            dcg += 1 / math.log2(i + 2)  # rank starts at 1
    ideal_dcg = sum([1 / math.log2(i + 2) for i in range(len(expected))])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

# -----------------------
# Process single query
# -----------------------
async def process_query(query: str, expected_ids: List[str]):
    try:
        predicted_ids = await get_predicted_file_ids_async(query)

        score_per_id = [1 if eid in predicted_ids else 0 for eid in expected_ids]
        avg_score = sum(score_per_id) / len(score_per_id) if score_per_id else 0

        precision, recall = precision_recall(predicted_ids, expected_ids)
        ap = average_precision(predicted_ids, expected_ids)
        rr = reciprocal_rank(predicted_ids, expected_ids)
        ndcg_score = ndcg(predicted_ids, expected_ids)

        print(f"Processed query '{query}' | Avg score: {round(avg_score,3)} | Precision: {precision:.3f}, Recall: {recall:.3f}")

        return {
            "query": query,
            "expected": expected_ids,
            "predicted": predicted_ids,
            "score_per_id": score_per_id,
            "average_score": round(avg_score, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "average_precision": round(ap, 3),
            "reciprocal_rank": round(rr, 3),
            "ndcg": round(ndcg_score, 3)
        }

    except Exception as e:
        print(f"Error processing query '{query}': {e}")
        return {
            "query": query,
            "expected": expected_ids,
            "error": str(e)
        }

# -----------------------
# Evaluate Excel endpoint
# -----------------------
@app.post("/evaluate-excel")
async def evaluate_excel(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")
        df = pd.read_excel(file.file, engine="openpyxl")

        if "query" not in df.columns or "expected_ids" not in df.columns:
            print("Excel missing required columns: 'query' and 'expected_ids'")
            raise HTTPException(
                status_code=400,
                detail="Excel must have 'query' and 'expected_ids' columns"
            )

        tasks = []
        for _, row in df.iterrows():
            query = row["query"]
            expected_ids = str(row["expected_ids"]).split(",")
            expected_ids = [eid.strip() for eid in expected_ids if eid.strip()]
            tasks.append(process_query(query, expected_ids))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate overall metrics
        def average_metric(results, key):
            vals = [r[key] for r in results if isinstance(r, dict) and key in r]
            return round(sum(vals)/len(vals), 3) if vals else 0

        overall_metrics = {
            "overall_average_score": average_metric(results, "average_score"),
            "precision": average_metric(results, "precision"),
            "recall": average_metric(results, "recall"),
            "MAP": average_metric(results, "average_precision"),
            "MRR": average_metric(results, "reciprocal_rank"),
            "NDCG": average_metric(results, "ndcg")
        }

        print(f"Finished processing Excel. Overall metrics: {overall_metrics}")

        return {
            "overall_metrics": overall_metrics,
            "details": results
        }

    except Exception as e:
        print(f"Error in /evaluate-excel: {e}")
        raise HTTPException(status_code=500, detail=str(e))





# @app.get("/list-drive-files/")
# def list_drive_files(folder_url: str):
#     """
#     Extract and return list of files (file_id + file_name) from a Google Drive folder.
#     """
#     try:
#         files = extract_file_ids_and_names(folder_url)
#         return JSONResponse(content={"count": len(files), "files": files})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})


# Define request format
class EvalItem(BaseModel):
    query: str

class EvalBatchRequest(BaseModel):
    queries: List[EvalItem]


