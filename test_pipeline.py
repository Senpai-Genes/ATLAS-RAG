
from query_processing.decomposer import QueryDecomposer
from retrieval.colbert_retriever import ColBERTRetriever
from retrieval.evidence_tracer import EvidenceTracer


def main():
    # Initialize components
    decomposer = QueryDecomposer()
    retriever = ColBERTRetriever(rrf_k=60)
    tracer = EvidenceTracer()

    # Original user query
    query = (
        "How did Project Aurora's forecasting models perform, "
        "and what improvements were introduced later?"
    )

    print("\n" + "=" * 60)
    print("ORIGINAL QUERY")
    print("=" * 60)
    print(query)

    # Step 1: Query decomposition
    decomposition = decomposer.decompose(query)
    subqueries = decomposition["subqueries"]

    print("\n" + "=" * 60)
    print("QUERY DECOMPOSITION")
    print("=" * 60)
    print("Decomposed:", decomposition["decomposed"])

    for index, subquery in enumerate(subqueries, start=1):
        print(f"Subquery {index}: {subquery}")

    # Step 2: ColBERT retrieval + RRF fusion
    results = retriever.retrieve(
        queries=subqueries,
        k=5
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS (RRF RANKING)")
    print("=" * 60)

    print("Total unique nodes:", len(results))

    # Step 3: Display ranked results
    for rank, result in enumerate(results, start=1):
        print(f"\n{'=' * 60}")
        print(f"RESULT {rank}")
        print("=" * 60)

        print("Node ID:", result["document_id"])
        print("RRF Score:", round(result["rrf_score"], 6))

        print("\nMatched Subqueries:")
        for matched_query in result["matched_queries"]:
            print("-", matched_query)

        print("\nQuery Rankings:")
        for matched_query, query_rank in result["query_ranks"].items():
            print(f"- {matched_query}: Rank {query_rank}")

        print("\nOriginal ColBERT Scores:")
        for matched_query, score in result["colbert_scores"].items():
            print(f"- {matched_query}: {score:.4f}")

        # Step 4: RAPTOR evidence tracing
        print("\nSOURCE EVIDENCE")

        evidence = tracer.trace(result["document_id"])

        if not evidence:
            print("No source evidence found.")
            continue

        for leaf in evidence:
            print("\nLeaf ID:", leaf["id"])
            print("Source:", leaf["source"])
            print("Page:", leaf["page"])
            print("Evidence:", leaf["text"][:400])

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
