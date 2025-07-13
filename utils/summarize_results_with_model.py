
from utils.llm_client import TextGenerator

def summarize_results_with_model(query: str, grouped_results: list, api_key: str) -> dict:
    import google.generativeai as genai
    print("Summarizing results with Model...")
    genai.configure(api_key=api_key)
    
    # Combine top chunks into a single context string
    context_chunks = []
    for result in grouped_results:
        for chunk in result.get("top_chunks", []):
            context_chunks.append(chunk.get("chunk", ""))

    context_text = "\n\n".join(context_chunks[:20])  # Limit to top 20 chunks

    prompt = f"""
You are a helpful assistant. Based on the following context, answer the user's question in **Markdown format**.

## User Question
{query}

## Context
{context_text}

## Guidelines


Please follow these strict formatting and response guidelines:
---

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

### 🧠 Content Guidelines
- Provide a clear, informative, and helpful answer based on the query and context.
- Break down complex concepts into easy-to-read sections.
- Avoid large blocks of text. Prefer short paragraphs with spacing.
- Add examples or real-world analogies when helpful.
- End with a `### Summary` section that highlights key takeaways in 2–4 bullet points.

---

### 💡 Tone
- Be concise, accurate, and easy to follow.
- Maintain a professional, friendly tone suitable for a knowledge base or educational UI.

## Markdown Answer
"""

    model = TextGenerator(
        api_key=api_key
    )
    response = model.run(prompt)

    return {
        "markdown": response.text
    }
