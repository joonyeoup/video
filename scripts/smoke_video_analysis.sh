#!/usr/bin/env bash
# Smoke test for the `analyze_video` pipeline.
#
# Runs the detection helper standalone (no VLM required), then optionally
# exercises the full tool through the ZeroClaw agent if a VLM base URL is set.
#
# Usage:
#   scripts/smoke_video_analysis.sh /path/to/sample.mp4 [models-dir]
#
# Environment:
#   PYTHON_BIN  Python interpreter to use (default: python3)
#   VLM_URL     If set, prints the agent command for the end-to-end run.

set -euo pipefail

VIDEO="${1:?usage: smoke_video_analysis.sh /path/to/sample.mp4 [models-dir]}"
MODELS_DIR="${2:-./models}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HELPER="$REPO_ROOT/crates/zeroclaw-tools/src/video_analysis_helper.py"

echo "== 1. Python + OpenCV available? =="
"$PYTHON_BIN" -c 'import cv2; print("opencv", cv2.__version__)' || {
  echo "FAIL: install with: $PYTHON_BIN -m pip install opencv-python-headless" >&2
  exit 1
}

echo "== 2. Model weights present? =="
if [ ! -s "$MODELS_DIR/face_detection_yunet_2023mar.onnx" ] \
   && [ ! -s "$MODELS_DIR/res10_300x300_ssd_iter_140000.caffemodel" ]; then
  echo "FAIL: no detector weights in $MODELS_DIR — run scripts/download_face_models.sh \"$MODELS_DIR\"" >&2
  exit 1
fi

echo "== 3. Frame sampling + face detection on $VIDEO =="
OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT
MANIFEST="$("$PYTHON_BIN" "$HELPER" \
  --video "$VIDEO" --fps 1.0 --detector yunet \
  --models-dir "$MODELS_DIR" --out-dir "$OUT_DIR/crops")"
echo "$MANIFEST" | "$PYTHON_BIN" -c '
import json, sys
m = json.load(sys.stdin)
print(f"detector: {m[\"detector_used\"]}  frames sampled: {m[\"frames_sampled\"]}  frames with faces: {m[\"frames_with_faces\"]}")
'
echo "face crops written: $(find "$OUT_DIR/crops" -name '*.jpg' | wc -l | tr -d ' ')"

echo "== 4. End-to-end via the agent =="
if [ -n "${VLM_URL:-}" ]; then
  echo "With [video_analysis] enabled in your config, run:"
else
  echo "Set video_analysis.vlm.base_url/model in your config, then run:"
fi
echo "  zeroclaw agent --message \"analyze the video at $VIDEO\""
echo "and check the report under your configured video_analysis.output_dir."
echo "SMOKE TEST PASSED (local pipeline)"
