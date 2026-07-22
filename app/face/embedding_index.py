from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TenantEmbeddingIndex:
    items: list[dict]
    matrix: np.ndarray


def build_embedding_index(items: list[dict]) -> TenantEmbeddingIndex:
    usable_items = []
    vectors = []
    expected_size = None
    for item in items:
        vector = np.asarray(item.get("embedding"), dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            continue
        if expected_size is None:
            expected_size = vector.size
        if vector.size != expected_size:
            continue
        usable_items.append(item)
        vectors.append(vector)

    matrix = (
        np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
        if vectors
        else np.empty((0, 0), dtype=np.float32)
    )
    matrix.setflags(write=False)
    return TenantEmbeddingIndex(items=usable_items, matrix=matrix)


def best_embedding_candidate(
    index: TenantEmbeddingIndex,
    detected_embedding: np.ndarray,
) -> dict | None:
    if index.matrix.size == 0 or index.matrix.shape[1] != detected_embedding.size:
        return None
    scores = index.matrix @ detected_embedding
    best_index = int(np.argmax(scores))
    item = index.items[best_index]
    return {
        "employeeId": item["employeeId"],
        "employeeName": item.get("employeeName"),
        "score": float(scores[best_index]),
    }
