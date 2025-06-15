from pymilvus import Collection
from vectorstore.milvus_client import MilvusClient

def search_chunks(query_embedding: list, top_k=20):
    milvus = MilvusClient()
    collection = Collection("pdf_chunks")
    collection.load()
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["pdf_id", "file_id", "chunk"]
    )
    
    hits = results[0]  # Only one query
    return [
        {
            "file_id": hit.entity.get("file_id"),
            "pdf_id": hit.entity.get("pdf_id"),
            "chunk": hit.entity.get("chunk"),
            "score": hit.distance
        }
        for hit in hits
    ]
