
from ragatouille import RAGPretrainedModel


class ColBERTRetriever:
    def __init__(
        self,
        index_path="data/processed/colbert/colbert/indexes/raptor_colbert",
        rrf_k=60,
    ):
        self.model = RAGPretrainedModel.from_index(index_path)
        self.rrf_k = rrf_k

    def retrieve(self, queries, k=5):
        if isinstance(queries, str):
            queries = [queries]

        queries = list(dict.fromkeys(queries))
        merged_results = {}

        for query in queries:
            results = self.model.search(query=query, k=k)

            seen_in_query = set()

            for rank, result in enumerate(results, start=1):
                node_id = str(result["document_id"])

                # Prevent duplicate contributions from one query.
                if node_id in seen_in_query:
                    continue

                seen_in_query.add(node_id)

                contribution = 1.0 / (self.rrf_k + rank)

                if node_id not in merged_results:
                    merged_results[node_id] = {
                        "document_id": node_id,
                        "content": result["content"],
                        "rrf_score": 0.0,
                        "matched_queries": [],
                        "query_ranks": {},
                        "colbert_scores": {},
                    }

                entry = merged_results[node_id]

                entry["rrf_score"] += contribution
                entry["matched_queries"].append(query)
                entry["query_ranks"][query] = rank
                entry["colbert_scores"][query] = float(
                    result["score"]
                )

        return sorted(
            merged_results.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )
