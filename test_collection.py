
from pipeline import RAGPipeline
from generation.context_builder import ContextBuilder
from generation.generator import AnswerGenerator


pipeline = RAGPipeline(collection_name="aurora_test")

query = (
    "How did Project Aurora's forecasting models "
    "perform, and what improvements were introduced later?"
)

print("\nRunning collection:", pipeline.collection_name)

result = pipeline.run(query, k=5)

context_builder = ContextBuilder()
context = context_builder.build(result["results"])

generator = AnswerGenerator()
answer = generator.generate(
    query=query,
    context=context["context"]
)

print("\n===== ANSWER =====")
print(answer)

print("\n===== SOURCES =====")
for source in context["sources"]:
    print(
        f"[{source['citation_id']}] "
        f"{source['source']} "
        f"(Page {source['page']})"
    )
