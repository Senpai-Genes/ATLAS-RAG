
import json
from pathlib import Path

from query_processing.router import QueryRouter
from query_processing.decomposer import QueryDecomposer
from retrieval.colbert_retriever import ColBERTRetriever
from retrieval.evidence_tracer import EvidenceTracer


class RAGPipeline:
    def __init__(
        self,
        collection_name=None,
        collections_root="data/collections",
        tree_path="data/processed/raptor_tree.json"
    ):
        self.router = QueryRouter()
        self.decomposer = QueryDecomposer()

        if collection_name is not None:
            collection_dir = (
                Path(collections_root) / collection_name
            ).resolve()

            metadata_path = collection_dir / "metadata.json"

            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Collection not found: {collection_name}"
                )

            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            tree_path = metadata["tree_path"]
            index_path = metadata["index_path"]

            self.collection_name = collection_name

        else:
            index_path = (
                "data/processed/colbert/colbert/"
                "indexes/raptor_colbert"
            )
            self.collection_name = "legacy"

        self.retriever = ColBERTRetriever(
            index_path=str(index_path)
        )
        self.tracer = EvidenceTracer(
            tree_path=str(tree_path)
        )

        with open(tree_path, "r", encoding="utf-8") as f:
            tree = json.load(f)

        self.nodes = {
            node["id"]: node for node in tree["nodes"]
        }

    def run(self, query, k=5):
        routing = self.router.route(query)
        route = routing["route"]

        if route == "decomposed":
            queries = self.decomposer.decompose(
                query
            )["subqueries"]
        else:
            queries = [query]

        search_k = (
            max(k * 3, 10)
            if route == "hierarchical"
            else k
        )

        results = self.retriever.retrieve(
            queries=queries,
            k=search_k
        )

        if route == "hierarchical":
            results = sorted(
                results,
                key=lambda result: (
                    self.nodes.get(
                        result["document_id"], {}
                    ).get("level", 0) > 0,
                    result["rrf_score"]
                ),
                reverse=True
            )

        results = results[:k]

        for result in results:
            node_id = result["document_id"]

            if node_id not in self.nodes:
                result["evidence"] = []
                continue

            result["level"] = self.nodes[node_id]["level"]
            result["evidence"] = self.tracer.trace(node_id)

        return {
            "query": query,
            "collection": self.collection_name,
            "route": route,
            "routing_reason": routing["reason"],
            "subqueries": queries,
            "results": results
        }
