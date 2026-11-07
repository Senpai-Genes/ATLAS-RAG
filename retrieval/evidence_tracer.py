import json


class EvidenceTracer:
    def __init__(self, tree_path="data/processed/raptor_tree.json"):
        with open(tree_path, "r", encoding="utf-8") as f:
            tree = json.load(f)

        self.nodes = {
            node["id"]: node
            for node in tree["nodes"]
        }

    def trace(self, node_id):
        """Return original leaf nodes supporting a RAPTOR node."""
        if node_id not in self.nodes:
            raise ValueError(f"Unknown node: {node_id}")

        evidence = []
        visited = set()

        def traverse(current_id):
            if current_id in visited:
                return

            visited.add(current_id)
            node = self.nodes[current_id]

            if node["level"] == 0:
                evidence.append({
                    "id": node["id"],
                    "text": node["text"],
                    "source": node["metadata"].get("source"),
                    "page": node["metadata"].get("page"),
                })
                return

            for child_id in node["children"]:
                traverse(child_id)

        traverse(node_id)
        return evidence