

import argparse

from indexing.ingestion_manager import IngestionManager


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into a RAG collection."
    )

    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF file"
    )

    parser.add_argument(
        "--collection",
        required=True,
        help="Unique collection name"
    )

    args = parser.parse_args()

    manager = IngestionManager()
    result = manager.ingest(
        pdf_path=args.pdf,
        collection_name=args.collection
    )

    print("\n===== COLLECTION DETAILS =====")
    print("Collection:", result["collection_name"])
    print("Pages:", result["pages_loaded"])
    print("Leaf chunks:", result["leaf_chunks"])
    print("Total RAPTOR nodes:", result["total_nodes"])
    print("Index:", result["index_path"])


if __name__ == "__main__":
    main()
