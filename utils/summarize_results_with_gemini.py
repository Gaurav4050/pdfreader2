def summarize_results_with_gemini(query: str, grouped_results: list, api_key: str) -> dict:
    import google.generativeai as genai
    print("Summarizing results with Gemini...")
    genai.configure(api_key=api_key)
    
    # Combine top chunks into a single context string
    context_chunks = []
    for result in grouped_results:
        for chunk in result.get("top_chunks", []):
            context_chunks.append(chunk.get("chunk", ""))

    context_text = "\n\n".join(context_chunks[:10])  # Limit to top 10 chunks

    prompt = f"""
You are a helpful assistant. Based on the following context, answer the user's question in **Markdown format**.

## User Question
{query}

## Context
{context_text}

## Guidelines
- Use bullet points for lists.
- Use headers if applicable.
- Highlight important terms with **bold**.
- Provide a clear and informative response.
- End with a brief summary if appropriate.

## Markdown Answer
"""

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)

    return {
        "markdown": response.text
    }
