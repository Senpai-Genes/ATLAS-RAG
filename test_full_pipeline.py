
from pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline()

    queries = [
        "What is Project Aurora?",
        (
            "How did Project Aurora's forecasting models perform, "
            "and what improvements were introduced later?"
        ),
        "Summarize all projects at Atlas Research Lab.",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print("QUERY:", query)

        output = pipeline.run(query, k=5)

        print("ROUTE:", output["route"])
        print("REASON:", output["routing_reason"])

        print("\nSUBQUERIES:")
        for subquery in output["subqueries"]:
            print("-", subquery)

        print("\nRESULTS:")

        for result in output["results"]:
            print("\nNode:", result["document_id"])
            print("Level:", result.get("level"))
            print("RRF:", round(result["rrf_score"], 6))

            for leaf in result["evidence"]:
                print(
                    "Source:",
                    leaf["source"],
                    "| Page:",
                    leaf["page"],
                )


if __name__ == "__main__":
    main()
