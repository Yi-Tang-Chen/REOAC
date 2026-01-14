#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-}"
SAMPLE_PATH="${2:-}"

if [[ -z "$RUN_DIR" ]]; then
  echo "Usage: $0 <run_dir> [sample_json]"
  exit 1
fi

VIZ_DIR="${RUN_DIR}/viz"
SAMPLES_DIR="${RUN_DIR}/samples"
METRICS_PATH="${RUN_DIR}/metrics.jsonl"

mkdir -p "$VIZ_DIR"

export RUN_DIR
export VIZ_DIR
export SAMPLES_DIR
export METRICS_PATH

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/third_party:${PYTHONPATH:-}"

if [[ -z "$SAMPLE_PATH" && -d "$SAMPLES_DIR" ]]; then
  SAMPLE_PATH="$(
    python - <<'PY'
import glob
import json
import os

samples = sorted(glob.glob(os.path.join(os.environ.get("SAMPLES_DIR", ""), "*.json")))
for path in samples:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        relation = payload.get("relation")
        if relation:
            print(path)
            break
    except Exception:
        continue
PY
  )"
fi

if [[ -n "$SAMPLE_PATH" && -f "$SAMPLE_PATH" ]]; then
  python -m src.viz.energy_maps --input "$SAMPLE_PATH" --output-dir "$VIZ_DIR"
  python - <<'PY'
import os

csv_path = os.path.join(os.environ["VIZ_DIR"], "energy_map.csv")
png_path = os.path.join(os.environ["VIZ_DIR"], "energy_map.png")
try:
    import numpy as np
    import matplotlib.pyplot as plt
except Exception:
    print("matplotlib not available; skipping energy_map.png")
    raise SystemExit(0)

data = np.loadtxt(csv_path, delimiter=",")
plt.imshow(data, aspect="auto", cmap="viridis")
plt.colorbar()
plt.title("Relation Energy Map")
plt.xlabel("Generated token index")
plt.ylabel("Prompt token index")
plt.tight_layout()
plt.savefig(png_path, dpi=200)
print(png_path)
PY
else
  echo "No sample with relation found; skipping energy map."
fi

python - <<'PY'
import glob
import json
import os

run_dir = os.environ["RUN_DIR"]
samples = sorted(glob.glob(os.path.join(run_dir, "samples", "*.json")))
features = []
for path in samples:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        vector = payload.get("feature_vector")
        if vector:
            features.append(vector)
    except Exception:
        continue

out_path = os.path.join(run_dir, "viz", "features.json")
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump({"features": features}, handle)
print(out_path)
PY

python -m src.viz.embedding_proj \
  --input "$VIZ_DIR/features.json" \
  --output "$VIZ_DIR/embedding_proj.csv"

python - <<'PY'
import csv
import os

csv_path = os.path.join(os.environ["VIZ_DIR"], "embedding_proj.csv")
png_path = os.path.join(os.environ["VIZ_DIR"], "embedding_proj.png")
try:
    import matplotlib.pyplot as plt
except Exception:
    print("matplotlib not available; skipping embedding_proj.png")
    raise SystemExit(0)

xs, ys = [], []
with open(csv_path, "r", encoding="utf-8") as handle:
    reader = csv.reader(handle)
    next(reader, None)
    for row in reader:
        if len(row) < 2:
            continue
        xs.append(float(row[0]))
        ys.append(float(row[1]))

if not xs:
    print("No features to plot; embedding_proj.png not generated.")
    raise SystemExit(0)

plt.scatter(xs, ys, s=6, alpha=0.7)
plt.title("Feature Embedding Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig(png_path, dpi=200)
print(png_path)
PY

if [[ -f "$METRICS_PATH" ]]; then
  python - <<'PY'
import json
import os

metrics_path = os.environ["METRICS_PATH"]
out_png = os.path.join(os.environ["VIZ_DIR"], "loss.png")
try:
    import matplotlib.pyplot as plt
except Exception:
    print("matplotlib not available; skipping loss.png")
    raise SystemExit(0)

iters, lq, lp, lb, lt = [], [], [], [], []
with open(metrics_path, "r", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        iters.append(row.get("iteration", len(iters)))
        lq.append(row.get("critic_loss", 0.0))
        lp.append(row.get("actor_loss", 0.0))
        lb.append(row.get("backbone_loss", 0.0))
        lt.append(row.get("total_loss", 0.0))

plt.plot(iters, lq, label="critic")
plt.plot(iters, lp, label="actor")
plt.plot(iters, lb, label="backbone")
plt.plot(iters, lt, label="total")
plt.legend()
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(out_png, dpi=200)
print(out_png)
PY
else
  echo "metrics.jsonl not found; skipping loss plot."
fi
