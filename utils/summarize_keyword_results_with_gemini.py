def summarize_keyword_results_with_gemini(query: str, matched_chunks: list, api_key: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    # Combine top 10 matched chunks
    context_text = "\n\n".join(chunk.get("chunk", "") for chunk in matched_chunks[:10])

    prompt = f"""
You are a helpful assistant. Based on the following text chunks, extract the **most relevant bullet points** related to the user's **keyword-based query** so just return the bullet points without any additional text do not add any additional text or explanation.

## User Keyword Query
"{query}"

## Relevant Chunks
{context_text}

## Guidelines
- Output should be in **Markdown format**
- List each important point as a bullet (`- `)
- Bold keywords or key phrases that match the query
- If possible, add sub-bullets for extra details
- Make it easy to read and scan

## Keyword-Based Highlights
"""

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)

    return {
        "markdown": response.text
    }
