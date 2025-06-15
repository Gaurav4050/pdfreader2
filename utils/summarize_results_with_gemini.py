def summarize_results_with_gemini(query: str, grouped_results: list, api_key: str):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    # Combine top chunks into a single context string
    context_chunks = []
    for result in grouped_results:
        for chunk in result["top_chunks"]:
            context_chunks.append(chunk["chunk"])
    
    context_text = "\n\n".join(context_chunks[:10])  # Limit to top 10 chunks
    
    prompt = f"""You are a helpful assistant. Based on the following information, answer the user's question.

    User Question: {query}

    Context:
    {context_text}

    Answer:"""

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text
