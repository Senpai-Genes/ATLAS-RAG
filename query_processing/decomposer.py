
import json

from ollama import Client


class QueryDecomposer:
    def __init__(
        self,
        model="qwen3:4b",
        host="http://localhost:11434",
        max_subqueries=4,
    ):
        self.client = Client(host=host)
        self.model = model
        self.max_subqueries = max_subqueries

    def decompose(self, query):
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        system_prompt = f"""
You are a query decomposition module for a RAG system.

Your task:
- Determine whether the user's question requires multiple
  independent information retrieval operations.
- Keep simple, single-topic questions unchanged.
- Split complex, multi-part questions into focused subqueries.
- Preserve important names, dates, and constraints.
- Do not answer the question.
- Do not invent information.
- Generate at most {self.max_subqueries} subqueries.

Return ONLY a valid JSON object with this structure:

{{
  "decomposed": true,
  "subqueries": [
    "First question?",
    "Second question?"
  ]
}}

For simple questions, return:

{{
  "decomposed": false,
  "subqueries": ["Original question"]
}}
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
            options={
                "temperature": 0,
                "num_predict": 400,
            },
        )

        raw_output = response.message.content.strip()

        try:
            result = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Model returned invalid JSON: {raw_output}"
            ) from exc

        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object.")

        subqueries = result.get("subqueries")

        if not isinstance(subqueries, list):
            raise ValueError("Expected a list of subqueries.")

        subqueries = [
            item.strip()
            for item in subqueries
            if isinstance(item, str) and item.strip()
        ]

        if not subqueries:
            subqueries = [query]

        subqueries = list(dict.fromkeys(subqueries))
        subqueries = subqueries[:self.max_subqueries]

        if len(subqueries) < 2:
            return {
                "decomposed": False,
                "subqueries": [query],
            }

        return {
            "decomposed": True,
            "subqueries": subqueries,
        }
