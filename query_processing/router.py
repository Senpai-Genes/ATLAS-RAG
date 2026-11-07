
import json
from ollama import Client


class QueryRouter:
    VALID_ROUTES = {"direct", "decomposed", "hierarchical"}

    def __init__(
        self,
        model="qwen3:4b",
        host="http://localhost:11434",
    ):
        self.client = Client(host=host)
        self.model = model

    def route(self, query):
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        system_prompt = """
You are a query router for an advanced RAG system.

Classify the user's question into exactly one route:

1. direct:
   Simple factual questions that can be answered using
   one focused retrieval query.

2. decomposed:
   Multi-part questions requiring several independent
   pieces of evidence.

3. hierarchical:
   Broad questions asking for an overview, summary,
   or explanation across multiple document sections.

Rules:
- Return exactly one route.
- Do not answer the question.
- Do not invent information.
- Output valid JSON only.

Output format:
{
  "route": "direct",
  "reason": "Brief explanation"
}
"""

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            format="json",
            think=False,
            options={"temperature": 0},
        )

        try:
            result = json.loads(response.message.content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Router returned invalid JSON."
            ) from exc

        route = result.get("route")

        if route not in self.VALID_ROUTES:
            raise ValueError(f"Invalid route: {route}")

        return {
            "route": route,
            "reason": result.get("reason", ""),
        }
