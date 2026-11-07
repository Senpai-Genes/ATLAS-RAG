
from ollama import Client


class RaptorSummarizer:
    def __init__(self, model="qwen3:4b"):
        self.client = Client(host="http://localhost:11434")
        self.model = model

    def summarize(self, chunks):
        if not chunks:
            raise ValueError("Cannot summarize an empty cluster")

        combined_text = "\n\n".join(
            f"[{chunk['id']}]\n{chunk['text']}"
            for chunk in chunks
        )

        system_prompt = """
You are a document summarization engine.

Produce factual summaries for a RAPTOR retrieval index.

Rules:
1. Summarize the document content directly.
2. Preserve names, dates, numbers and technical details.
3. Do not describe the task or instructions.
4. Do not include reasoning, planning or analysis.
5. Do not invent facts.
6. Keep unrelated topics separate.
7. Output only the final factual summary.
"""

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize the following source excerpts:\n\n"
                        + combined_text
                    )
                }
            ],
            think=False,
            options={
                "temperature": 0,
                "num_predict": 700
            }
        )

        summary = (response.message.content or "").strip()

        if not summary:
            raise RuntimeError("LLM returned an empty summary")

        # Reject common reasoning-style outputs
        invalid_starts = (
            "hmm,",
            "okay,",
            "let me think",
            "the user wants",
            "we need to summarize"
        )

        if summary.lower().startswith(invalid_starts):
            raise RuntimeError(
                "LLM returned reasoning instead of a summary: "
                + summary[:150]
            )

        return summary
