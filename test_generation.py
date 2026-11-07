
from pipeline import RAGPipeline
from generation.context_builder import ContextBuilder
from generation.generator import AnswerGenerator


def main():
    pipeline = RAGPipeline()
    context_builder = ContextBuilder()
    generator = AnswerGenerator()

    query = (
        "How did Project Aurora's forecasting models perform, "
        "and what improvements were introduced later?"
    )

    # Retrieve evidence
    retrieval = pipeline.run(query, k=5)

    # Build source context
    context_data = context_builder.build(
        retrieval["results"]
    )

    # Generate answer
    answer = generator.generate(
        query=query,
        context=context_data["context"],
    )

    print("\n" + "=" * 60)
    print("QUESTION:", query)
    print("ROUTE:", retrieval["route"])

    print("\nGENERATED ANSWER:\n")
    print(answer)

    print("\nSOURCE REFERENCES:\n")

    for source in context_data["sources"]:
        print(
            f"[{source['citation_id']}] "
            f"{source['source']} "
            f"(Page {source['page']}, "
            f"Leaf {source['leaf_id']})"
        )


if __name__ == "__main__":
    main()
