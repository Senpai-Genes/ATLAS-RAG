
from ollama import Client


class QueryRewriter:
    def __init__(
        self,
        model="qwen3:4b",
        host="http://localhost:11434",
    ):
        self.client = Client(host=host)
        self.model = model

    def rewrite(self, question, history):
        if not history:
            return question

        conversation = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in history[-12:]
        )

        prompt = f"""
Rewrite the latest user question into a
self-contained question for document retrieval.

Use the conversation history only to resolve
references such as "it", "that", "the second one",
or "explain further".

Rules:
- Preserve the user's original intention.
- Do not answer the question.
- Do not introduce unsupported facts.
- If already standalone, return it unchanged.

Conversation:
{conversation}

Latest question:
{question}
"""

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            think=False,
            format={
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"],
            },
            options={"temperature": 0},
        )

        import json

        try:
            data = json.loads(response.message.content)
            rewritten = data.get("question", "").strip()
            return rewritten or question
        except (ValueError, TypeError, AttributeError):
            return question
