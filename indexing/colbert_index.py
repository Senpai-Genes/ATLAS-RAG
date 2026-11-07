
import json
from pathlib import Path

from ragatouille import RAGPretrainedModel


def build_colbert_index(
    tree_path="data/processed/raptor_tree.json",
    index_root="data/processed/colbert",
    index_name="raptor_colbert"
):
    tree_path = Path(tree_path)
    index_root = Path(index_root)

    if not tree_path.is_file():
        raise FileNotFoundError(
            f"RAPTOR tree not found: {tree_path}"
        )

    with tree_path.open("r", encoding="utf-8") as f:
        tree = json.load(f)

    nodes = tree["nodes"]

    if not nodes:
        raise ValueError("RAPTOR tree contains no nodes.")

    documents = [node["text"] for node in nodes]
    document_ids = [node["id"] for node in nodes]

    metadata = [
        {
            "node_id": node["id"],
            "level": node["level"]
        }
        for node in nodes
    ]

    index_root.mkdir(parents=True, exist_ok=True)

    print(f"Indexing {len(documents)} RAPTOR nodes")

    model = RAGPretrainedModel.from_pretrained(
        "colbert-ir/colbertv2.0",
        index_root=str(index_root)
    )

    index_path = model.index(
        index_name=index_name,
        collection=documents,
        document_ids=document_ids,
        document_metadatas=metadata,
        max_document_length=256,
        split_documents=False,
        overwrite_index=True
    )

    print("ColBERT index created:", index_path)

    return str(index_path)


if __name__ == "__main__":
    build_colbert_index()
