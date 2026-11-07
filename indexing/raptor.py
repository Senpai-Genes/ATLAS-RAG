
import json
from pathlib import Path

from indexing.embedder import RaptorEmbedder
from indexing.dimensionality import reduce_dimensions
from indexing.clustering import cluster_embeddings
from indexing.summarizer import RaptorSummarizer


class RaptorTreeBuilder:

    def __init__(
        self,
        max_depth=3,
        max_clusters=5,
        threshold=0.2,
        min_cluster_size=2
    ):
        self.max_depth = max_depth
        self.max_clusters = max_clusters
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size

        self.embedder = RaptorEmbedder()
        self.summarizer = RaptorSummarizer()

        self.all_nodes = []
        self.levels = {}

    def build(self, chunks):
        if not chunks:
            raise ValueError("No chunks provided")

        # Reset state for each build
        self.all_nodes = []
        self.levels = {}

        # Level 0: original leaf nodes
        current_nodes = []

        for chunk in chunks:
            node = {
                "id": chunk["id"],
                "text": chunk["text"],
                "level": 0,
                "children": [],
                "metadata": chunk["metadata"].copy()
            }
            current_nodes.append(node)

        self.all_nodes.extend(current_nodes)
        self.levels[0] = current_nodes

        # Recursive hierarchy construction
        for level in range(1, self.max_depth + 1):

            print(f"\n===== BUILDING LEVEL {level} =====")

            if len(current_nodes) < self.min_cluster_size:
                print("Stopping: insufficient nodes")
                break

            # Embed current-level nodes
            embeddings = self.embedder.embed(current_nodes)

            # UMAP needs enough samples
            if len(current_nodes) < 4:
                print("Stopping: too few nodes for UMAP")
                break

            reduced = reduce_dimensions(embeddings)

            # Prevent singleton-heavy clustering
            effective_max_clusters = min(
                self.max_clusters,
                max(1, len(current_nodes) // self.min_cluster_size)
            )

            assignments = cluster_embeddings(
                reduced,
                max_clusters=effective_max_clusters,
                threshold=self.threshold
            )

            groups = {}

            for node, cluster_ids in zip(
                current_nodes, assignments
            ):
                for cluster_id in cluster_ids:
                    groups.setdefault(
                        int(cluster_id), []
                    ).append(node)

            next_nodes = []

            for cluster_id, group in sorted(groups.items()):

                if len(group) < self.min_cluster_size:
                    print(
                        f"Skipping singleton cluster {cluster_id}"
                    )
                    continue

                # Do not create a parent identical in
                # coverage to the entire current level.
                if len(group) == len(current_nodes):
                    print(
                        "Stopping: cluster covers entire level"
                    )
                    continue

                print(
                    f"Summarizing cluster {cluster_id} "
                    f"with {len(group)} nodes"
                )

                summary = self.summarizer.summarize(group)

                parent = {
                    "id": f"L{level}_C{cluster_id}",
                    "text": summary,
                    "level": level,
                    "children": [
                        node["id"] for node in group
                    ],
                    "metadata": {
                        "cluster_id": cluster_id,
                        "child_count": len(group)
                    }
                }

                next_nodes.append(parent)

            if not next_nodes:
                print("Stopping: no valid parent nodes")
                break

            self.all_nodes.extend(next_nodes)
            self.levels[level] = next_nodes

            print(
                f"Level {level}: "
                f"{len(next_nodes)} parent nodes created"
            )

            # Use summaries as input to the next level
            current_nodes = next_nodes

        return self.all_nodes

    def save(self, path="data/processed/raptor_tree.json"):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "nodes": self.all_nodes,
                    "levels": self.levels
                },
                file,
                indent=2,
                ensure_ascii=False
            )

        print(f"\nRAPTOR tree saved to {output}")
