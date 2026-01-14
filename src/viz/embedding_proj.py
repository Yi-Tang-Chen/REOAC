"""Project feature vectors for visualization."""

from __future__ import annotations

import argparse
import csv
import json
import math
from typing import List


def _project_with_numpy(vectors: List[List[float]]) -> List[List[float]]:
    import numpy as np

    data = np.array(vectors, dtype=float)
    data = data - data.mean(axis=0, keepdims=True)
    cov = np.cov(data, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top = eigvecs[:, -2:]
    projected = data.dot(top)
    return projected.tolist()


def _project_fallback(vectors: List[List[float]]) -> List[List[float]]:
    projected = []
    for vec in vectors:
        if len(vec) >= 2:
            projected.append([vec[0], vec[1]])
        elif vec:
            projected.append([vec[0], 0.0])
        else:
            projected.append([0.0, 0.0])
    return projected


def project_vectors(vectors: List[List[float]]) -> List[List[float]]:
    try:
        return _project_with_numpy(vectors)
    except Exception:
        return _project_fallback(vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project embeddings")
    parser.add_argument("--input", required=True, help="JSON file with 'features' list")
    parser.add_argument("--output", default="embedding_proj.csv")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    vectors = payload.get("features") or []
    projected = project_vectors(vectors)

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y"])
        for point in projected:
            writer.writerow([f"{point[0]:.6f}", f"{point[1]:.6f}"])


if __name__ == "__main__":
    main()
