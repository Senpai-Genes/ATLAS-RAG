

class ContextBuilder:
    def __init__(self, max_chars=12000):
        self.max_chars = max_chars

    def build(self, results):
        seen = set()
        sources = []
        blocks = []
        total_chars = 0

        for result in results:
            for leaf in result.get("evidence", []):
                leaf_id = leaf["id"]

                if leaf_id in seen:
                    continue

                seen.add(leaf_id)

                text = leaf["text"].strip()
                source = leaf.get("source", "Unknown")
                page = leaf.get("page", "Unknown")

                citation_id = f"S{len(sources) + 1}"

                block = (
                    f"[{citation_id}]\n"
                    f"Source: {source}\n"
                    f"Page: {page}\n"
                    f"Content: {text}\n"
                )

                if total_chars + len(block) > self.max_chars:
                    continue

                blocks.append(block)
                total_chars += len(block)

                sources.append({
                    "citation_id": citation_id,
                    "leaf_id": leaf_id,
                    "source": source,
                    "page": page,
                    "text": text,
                })

        return {
            "context": "\n---\n".join(blocks),
            "sources": sources,
        }
