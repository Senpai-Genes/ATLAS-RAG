
from indexing.document_loader import load_pdf
from indexing.chunker import chunk_documents
from indexing.raptor import RaptorTreeBuilder


# STEP 1: Load document
documents = load_pdf(
    "data/raw/advanced_rag_sample.pdf"
)

print(f"Pages loaded: {len(documents)}")


# STEP 2: Create leaf chunks
chunks = chunk_documents(
    documents,
    chunk_size=150,
    overlap=30
)

print(f"Leaf nodes created: {len(chunks)}")


# STEP 3: Build RAPTOR hierarchy
raptor = RaptorTreeBuilder(
    max_depth=3,
    max_clusters=5,
    threshold=0.2,
    min_cluster_size=2
)

tree = raptor.build(chunks)


# STEP 4: Display hierarchy
print("\n===== RAPTOR TREE =====")

for level, nodes in sorted(raptor.levels.items()):

    print(f"\nLEVEL {level}")
    print("-" * 50)

    for node in nodes:
        print(f"Node: {node['id']}")

        if node["children"]:
            print(f"Children: {node['children']}")

        print(f"Text: {node['text'][:200]}...")
        print()


# STEP 5: Save the tree
raptor.save()


# STEP 6: Statistics
print("\n===== TREE STATISTICS =====")

print(f"Total nodes: {len(tree)}")
print(f"Total levels: {len(raptor.levels)}")

for level, nodes in sorted(raptor.levels.items()):
    print(f"Level {level}: {len(nodes)} nodes")
