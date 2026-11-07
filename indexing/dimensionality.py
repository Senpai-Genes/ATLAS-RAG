
import numpy as np
import umap


def reduce_dimensions(embeddings, n_components=5):
    n_samples = len(embeddings)

    if n_samples < 4:
        raise ValueError(
            "At least 4 embeddings are needed for this UMAP setup."
        )

    reducer = umap.UMAP(
        n_neighbors=min(10, n_samples - 1),
        n_components=min(n_components, n_samples - 2),
        metric="cosine",
        random_state=42
    )

    reduced_embeddings = reducer.fit_transform(embeddings)

    return reduced_embeddings
