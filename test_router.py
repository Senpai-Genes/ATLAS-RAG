
from query_processing.router import QueryRouter


def main():
    router = QueryRouter()

    queries = [
        "What is MAPE?",
        (
            "How did Project Aurora's forecasting models "
            "perform, and what improvements were introduced later?"
        ),
        "Summarize all projects at Atlas Research Lab.",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print("Query:", query)

        result = router.route(query)

        print("Route:", result["route"])
        print("Reason:", result["reason"])


if __name__ == "__main__":
    main()
