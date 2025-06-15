import pymupdf

def extract_text_from_pdf(path):
    try:
        doc = pymupdf.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from {path}: {e}")
        return ""
