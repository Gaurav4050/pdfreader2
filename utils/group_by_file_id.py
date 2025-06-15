from collections import defaultdict

def group_by_file_id(results: list, top_n=5):
    grouped = defaultdict(list)
    
    for result in results:
        grouped[result['file_id']].append(result)
    
    ranked = sorted(
        grouped.items(),
        key=lambda x: sum(r['score'] for r in x[1]) / len(x[1])  # avg distance
    )

    return [
        {
            "file_id": file_id,
            "top_chunks": sorted(chunks, key=lambda x: x['score'])[:2],
            "avg_score": sum(c['score'] for c in chunks) / len(chunks)
        }
        for file_id, chunks in ranked[:top_n]
    ]
