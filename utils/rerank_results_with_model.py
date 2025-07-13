from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import google.generativeai as genai
import traceback
from utils.llm_client import TextGenerator 

import traceback

def rerank_chunk(query: str, chunk: dict, model) -> dict | None:
    try:
        content = chunk.get("chunk", "")
        if not content:
            return None

        prompt = (
            "You are a domain expert ranking how well a content chunk answers a given user query. "
            "Give a score between 0 and 100 based on relevance, clarity, completeness, and accuracy.\n\n"
            "If the content contains exact keywords, phrases, or concepts from the query, treat it as more relevant. "
            "Reward content that directly matches or closely reflects the user's question. Penalize vague or unrelated text.\n\n"
            "Scoring Guide:\n"
            "- 100: Exact and complete answer. Highly relevant with key query terms clearly addressed.\n"
            "- 70-99: Mostly relevant. Key terms are present but may lack full context.\n"
            "- 40-69: Partially relevant. Mentions some related ideas but lacks clarity or depth.\n"
            "- 10-39: Slightly relevant. Vague overlap with the query.\n"
            "- 0-9: Completely irrelevant.\n\n"
            "Only return a single integer score between 0 and 100 with no explanation or extra text.\n\n"
            f"Query:\n{query}\n\n"
            f"Content:\n{content}\n\n"
            "Score (0-100):"
        )

        response = model.run(prompt)

        if hasattr(response, "text") and response.text:
            response_text = response.text.strip()
            digits = "".join(filter(str.isdigit, response_text))
            if digits:
                score = int(digits)
                score = max(0, min(score, 100))

                # Optional: Apply minor lexical boost if key query terms are present in content
                query_terms = set(query.lower().split())
                content_terms = set(content.lower().split())
                overlap = query_terms.intersection(content_terms)

                if len(overlap) >= 3:  # tweak threshold as needed
                    score = min(score + 10, 100)  # soft boost
                elif len(overlap) >= 1:
                    score = min(score + 5, 100)

                chunk["score"] = score
                return chunk
            else:
                chunk["score"] = 0
                return chunk
        else:
            return None
    except Exception as e:
        print(f"Error reranking chunk {chunk.get('file_id', 'N/A')}: {e}")
        import traceback
        traceback.print_exc()
        return None

def rerank_results_with_model_parallel(query: str, results: list[dict], api_key: str, top_k=10, max_workers=20) -> list[dict]:
    print(f"🔍 Parallel reranking {len(results)} chunks...")

    model = TextGenerator(
        api_key=api_key
    )

    reranked = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(rerank_chunk, query, chunk, model): chunk for chunk in results}

        for future in as_completed(future_to_chunk):
            result = future.result()
            if result is not None:
                reranked.append(result)

    return sorted(reranked, key=lambda x: x["score"], reverse=True)[:top_k]
