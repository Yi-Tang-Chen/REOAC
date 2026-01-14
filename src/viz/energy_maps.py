"""Visualization utilities for relational energy maps."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List


def save_energy_map(relation: List[List[float]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in relation:
            writer.writerow([f"{value:.6f}" for value in row])


def render_from_json(input_path: str, output_dir: str) -> str:
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    relation = payload.get("relation") or []
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "energy_map.csv")
    save_energy_map(relation, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render energy maps")
    parser.add_argument("--input", required=True, help="Path to JSON file containing relation matrix")
    parser.add_argument("--output-dir", default="runs", help="Directory to write the map")
    args = parser.parse_args()
    output_path = render_from_json(args.input, args.output_dir)
    print(output_path)


if __name__ == "__main__":
    main()
