
from pathlib import Path
import fitz


def load_pdf(file_path: str) -> list[dict]:
    documents = []

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append({
                "text": text,
                "metadata": {
                    "source": Path(file_path).name,
                    "page": page_number,
                }
            })

    return documents
