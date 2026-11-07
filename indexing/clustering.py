
import numpy as np
from sklearn.mixture import GaussianMixture


def cluster_embeddings(
    embeddings,
    max_clusters=5,
    threshold=0.2
):
    n_samples = len(embeddings)

    if n_samples < 2:
        return [[0] for _ in range(n_samples)]

    # Find a suitable cluster count using BIC
    best_bic = float("inf")
    best_model = None

    for n_clusters in range(1, min(max_clusters, n_samples) + 1):
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            reg_covar=1e-4,
            random_state=42
        )

        model.fit(embeddings)
        bic = model.bic(embeddings)

        if bic < best_bic:
            best_bic = bic
            best_model = model

    # Membership probabilities for each chunk
    probabilities = best_model.predict_proba(embeddings)

    clusters = []

    for row in probabilities:
        # Allow multiple cluster memberships
        memberships = np.where(row >= threshold)[0].tolist()

        # Ensure every chunk belongs to at least one cluster
        if not memberships:
            memberships = [int(np.argmax(row))]

        clusters.append(memberships)

    return clusters
