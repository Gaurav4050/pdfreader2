from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
import traceback

def rerank_chunk(query: str, chunk: dict, model) -> dict | None:
    try:
        content = chunk.get("chunk", "")
        if not content:
            return None

        prompt = f"Query: {query}\nContent: {content}\nHow relevant (0-100) is this content to the query?"
        response = model.generate_content(prompt)

        if hasattr(response, "text") and response.text:
            response_text = response.text.strip()
            score = int("".join(filter(str.isdigit, response_text)))
            chunk["score"] = score
            return chunk
        else:
            return None
    except Exception as e:
        print(f"Error reranking chunk {chunk.get('file_id', 'N/A')}: {e}")
        traceback.print_exc()
        return None


def rerank_results_with_gemini_parallel(query: str, results: list[dict], api_key: str, top_k=10, max_workers=20) -> list[dict]:
    print(f"🔍 Parallel reranking {len(results)} chunks...")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")

    reranked = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(rerank_chunk, query, chunk, model): chunk for chunk in results}

        for future in as_completed(future_to_chunk):
            result = future.result()
            if result is not None:
                reranked.append(result)

    return sorted(reranked, key=lambda x: x["score"], reverse=True)[:top_k]
