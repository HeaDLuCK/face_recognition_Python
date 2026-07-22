import unittest

import numpy as np

from app.face.embedding_index import best_embedding_candidate, build_embedding_index


class EmbeddingIndexTests(unittest.TestCase):
    def test_vectorized_match_returns_same_highest_dot_product(self) -> None:
        items = [
            {"employeeId": "A", "employeeName": "Alpha", "embedding": [1.0, 0.0, 0.0]},
            {"employeeId": "B", "employeeName": "Beta", "embedding": [0.0, 1.0, 0.0]},
        ]
        index = build_embedding_index(items)

        result = best_embedding_candidate(
            index,
            np.asarray([0.1, 0.9, 0.0], dtype=np.float32),
        )

        self.assertEqual(result["employeeId"], "B")
        self.assertAlmostEqual(result["score"], 0.9, places=6)

    def test_invalid_and_mismatched_embeddings_are_not_indexed(self) -> None:
        index = build_embedding_index(
            [
                {"employeeId": "A", "embedding": [1.0, 0.0]},
                {"employeeId": "EMPTY", "embedding": []},
                {"employeeId": "WRONG_SIZE", "embedding": [1.0, 0.0, 0.0]},
            ]
        )

        self.assertEqual([item["employeeId"] for item in index.items], ["A"])
        self.assertEqual(index.matrix.shape, (1, 2))
        self.assertIsNone(
            best_embedding_candidate(index, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        )


if __name__ == "__main__":
    unittest.main()
