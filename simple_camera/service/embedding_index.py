from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbeddingIndex:
    items: list[dict]
    matrix: np.ndarray


def build_embedding_index(items: list[dict]) -> EmbeddingIndex:
    usable_items: list[dict] = []
    vectors: list[np.ndarray] = []
    expected_size: int | None = None

    for item in items:
        vector = np.asarray(
            item.get("embedding"),
            dtype=np.float32,
        )

        if vector.ndim != 1 or vector.size == 0:
            continue

        norm = np.linalg.norm(vector)

        if norm == 0:
            continue

        # Normalize so matrix multiplication gives cosine similarity.
        vector = vector / norm

        if expected_size is None:
            expected_size = vector.size

        if vector.size != expected_size:
            continue

        usable_items.append(item)
        vectors.append(vector)

    if vectors:
        matrix = np.ascontiguousarray(
            np.stack(vectors),
            dtype=np.float32,
        )
    else:
        matrix = np.empty(
            (0, 0),
            dtype=np.float32,
        )

    matrix.setflags(write=False)

    return EmbeddingIndex(
        items=usable_items,
        matrix=matrix,
    )


def best_embedding_candidate(
    index: EmbeddingIndex,
    detected_embedding: np.ndarray,
    threshold: float = 0.50,
) -> dict | None:
    if index.matrix.size == 0:
        return None

    query = np.asarray(
        detected_embedding,
        dtype=np.float32,
    )

    if query.ndim != 1 or query.size == 0:
        return None

    if index.matrix.shape[1] != query.size:
        return None

    norm = np.linalg.norm(query)

    if norm == 0:
        return None

    query = query / norm

    scores = index.matrix @ query

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    if best_score < threshold:
        return None

    item = index.items[best_index]

    return {
        "employeeId": item["employeeId"],
        "employeeName": item.get("employeeName"),
        "embeddingId": item.get("embeddingId"),
        "etsAuth": item.get("etsAuth"),
        "score": best_score,
    }