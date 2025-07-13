# llm_client.py
import google.generativeai as genai

class TextGenerator:
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def run(self, prompt: str) -> str:
        result = self._model.generate_content(prompt)
        return result
