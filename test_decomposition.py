
from query_processing.decomposer import QueryDecomposer


def main():
    decomposer = QueryDecomposer()

    queries = [
        "What is MAPE?",
        "What is Project Aurora about?",
        (
            "How did Project Aurora's forecasting models "
            "perform, and what improvements were introduced later?"
        ),
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print("Original query:", query)

        result = decomposer.decompose(query)

        print("Decomposed:", result["decomposed"])

        for index, subquery in enumerate(
            result["subqueries"], start=1
        ):
            print(f"Subquery {index}: {subquery}")


if __name__ == "__main__":
    main()
