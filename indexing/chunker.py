
def chunk_documents(documents, chunk_size=500, overlap=100):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")

    chunks = []
    chunk_id = 0

    for document in documents:
        words = document["text"].split()
        step = chunk_size - overlap

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                continue

            chunks.append({
                "id": f"leaf_{chunk_id}",
                "text": " ".join(chunk_words),
                "metadata": document["metadata"].copy(),
                "level": 0
            })

            chunk_id += 1

            if start + chunk_size >= len(words):
                break

    return chunks
