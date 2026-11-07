
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from indexing.document_loader import load_pdf
from indexing.chunker import chunk_documents
from indexing.raptor import RaptorTreeBuilder
from indexing.colbert_index import build_colbert_index


class IngestionManager:
    def __init__(self, collections_root="data/collections"):
        self.collections_root = Path(collections_root)
        self.collections_root.mkdir(parents=True, exist_ok=True)

    def ingest(self, pdf_path, collection_name):
        pdf_path = Path(pdf_path).expanduser().resolve()

        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        if not re.fullmatch(r"[A-Za-z0-9_-]+", collection_name):
            raise ValueError(
                "Collection name must contain only letters, "
                "numbers, underscores, or hyphens."
            )

        collection_dir = self.collections_root / collection_name

        if collection_dir.exists():
            raise FileExistsError(
                f"Collection already exists: {collection_dir}"
            )

        collection_dir.mkdir(parents=True)
        documents_dir = collection_dir / "documents"
        documents_dir.mkdir()

        try:
            stored_pdf = documents_dir / pdf_path.name
            shutil.copy2(pdf_path, stored_pdf)

            print("\n[1/5] Loading PDF")
            documents = load_pdf(str(stored_pdf))

            if not documents:
                raise ValueError(
                    "No extractable text found in this PDF. "
                    "Scanned PDFs require OCR."
                )

            print(f"Pages loaded: {len(documents)}")

            print("\n[2/5] Chunking document")
            chunks = chunk_documents(
                documents,
                chunk_size=150,
                overlap=30
            )

            if not chunks:
                raise ValueError("No text chunks were generated.")

            print(f"Leaf chunks: {len(chunks)}")

            print("\n[3/5] Building RAPTOR hierarchy")
            raptor = RaptorTreeBuilder(
                max_depth=3,
                max_clusters=5,
                threshold=0.2,
                min_cluster_size=2
            )

            tree = raptor.build(chunks)
            tree_path = collection_dir / "raptor_tree.json"
            raptor.save(str(tree_path))

            print("\n[4/5] Building ColBERT index")
            index_path = build_colbert_index(
                tree_path=str(tree_path),
                index_root=str(collection_dir / "colbert"),
                index_name="raptor_colbert"
            )

            print("\n[5/5] Saving collection metadata")
            metadata = {
                "collection_name": collection_name,
                "original_filename": pdf_path.name,
                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "pdf_path": str(stored_pdf),
                "tree_path": str(tree_path),
                "index_path": str(index_path),
                "pages_loaded": len(documents),
                "leaf_chunks": len(chunks),
                "total_nodes": len(tree),
                "levels": len(raptor.levels)
            }

            with (collection_dir / "metadata.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(metadata, file, indent=2)

            print("\nIngestion completed successfully.")
            return metadata

        except Exception:
            shutil.rmtree(collection_dir)
            raise
