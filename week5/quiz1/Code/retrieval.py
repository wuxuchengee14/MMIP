"""Multi-positive class retrieval metrics; identical image IDs are excluded."""
import numpy as np


def retrieval_metrics(queries, gallery, query_labels, gallery_labels, query_ids, gallery_ids, ks=(1,5,10)):
    queries, gallery = np.asarray(queries), np.asarray(gallery)
    if not np.isfinite(queries).all() or not np.isfinite(gallery).all():raise ValueError("Nonfinite retrieval embeddings")
    queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    similarity = queries @ gallery.T
    query_ids, gallery_ids = np.asarray(query_ids), np.asarray(gallery_ids)
    valid = query_ids[:,None] != gallery_ids[None]
    similarity[~valid] = -np.inf
    ranking = np.argsort(-similarity, axis=1, kind="stable")
    relevance = (np.asarray(query_labels)[:,None] == np.asarray(gallery_labels)[None]) & valid
    positive_counts = relevance.sum(1)
    if (positive_counts == 0).any():
        raise ValueError("Every query requires a relevant, different gallery image")
    ranked_valid = np.take_along_axis(valid, ranking, axis=1)
    relevant = np.take_along_axis(relevance, ranking, axis=1)
    precision = relevant.cumsum(1) / np.arange(1, len(gallery)+1)[None]
    ap = (precision * relevant).sum(1) / positive_counts
    result = {"queries": len(queries), "gallery": len(gallery), "map": float(ap.mean())}
    for k in ks:
        limit = min(k, len(gallery))
        selected = relevant[:, :limit]
        valid_count = ranked_valid[:, :limit].sum(1)
        if (valid_count == 0).any():
            raise ValueError("No eligible gallery candidates")
        result[f"hit_at_{k}"] = float(selected.any(1).mean())
        result[f"precision_at_{k}"] = float((selected.sum(1) / valid_count).mean())
        result[f"recall_at_{k}"] = float((selected.sum(1) / positive_counts).mean())
        result[f"map_at_{k}"] = float(((precision[:,:limit]*selected).sum(1) / np.minimum(positive_counts,limit)).mean())
    return result, ranking, similarity, ap
