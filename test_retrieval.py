
from ragatouille import RAGPretrainedModel

INDEX_PATH = "data/processed/colbert/colbert/indexes/raptor_colbert"

print("Loading existing ColBERT index...")

model = RAGPretrainedModel.from_index(INDEX_PATH)

query = "What is Project Aurora about?"

print(f"\nQuery: {query}")

results = model.search(query=query, k=3)

for rank, result in enumerate(results, start=1):
    print(f"\n===== RESULT {rank} =====")
    print("Document ID:", result["document_id"])
    print("Score:", result["score"])
    print("Text:", result["content"][:500])

from retrieval.evidence_tracer import EvidenceTracer

tracer = EvidenceTracer()

for result in results:
    node_id = result["document_id"]

    print(f"\nRetrieved node: {node_id}")

    evidence = tracer.trace(node_id)

    for leaf in evidence:
        print(f"\nSource: {leaf['source']}")
        print(f"Page: {leaf['page']}")
        print(f"Leaf ID: {leaf['id']}")
        print(f"Evidence: {leaf['text'][:300]}")