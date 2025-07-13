from utils.llm_client import TextGenerator

def summarize_keyword_results_with_model(query: str, matched_chunks: list, api_key: str) -> dict:

    # Combine top 20 matched chunks
    context_text = "\n\n".join(chunk.get("chunk", "") for chunk in matched_chunks[:20])

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


### 📝 Markdown Formatting Rules
- Use `###` for main headings (e.g., topic title or section headers).
- Use `####` for subheadings (e.g., concept names or subsections).
- Separate each major section with a horizontal line (`---`) for clarity.
- Use **bullet points** (`-`) or numbered lists (`1.`, `2.`) for key points.
- Use **bold** (`**term**`) for highlighting important concepts.
- Use *italics* for emphasis or subtle notes.
- Use blockquotes (`>`) for examples or explanations.
- Use tables if needed for structured data or comparisons.
- Leave **empty lines** between paragraphs and sections for readability.

---

### 💡 Tone
- Be concise, accurate, and easy to follow.
- Maintain a professional, friendly tone suitable for a knowledge base or educational UI

## Keyword-Based Highlights
"""

    model = TextGenerator(
        api_key=api_key
    )
    response = model.run(prompt)

    return {
        "markdown": response.text
    }
