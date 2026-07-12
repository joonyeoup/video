#!/usr/bin/env bash
# Download face-detection model weights for the `analyze_video` tool.
#
# Fetches the YuNet ONNX detector (preferred) and the ResNet-SSD Caffe
# detector (fallback) into the directory given as $1, defaulting to ./models
# (the default of `video_analysis.models_dir`).
#
# Usage: scripts/download_face_models.sh [models-dir]

set -euo pipefail

MODELS_DIR="${1:-./models}"
mkdir -p "$MODELS_DIR"

YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SSD_PROTOTXT_URL="https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
SSD_WEIGHTS_URL="https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

fetch() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    echo "already present: $dest"
    return 0
  fi
  echo "downloading $(basename "$dest") ..."
  curl -fsSL --retry 3 -o "$dest" "$url"
}

fetch "$YUNET_URL" "$MODELS_DIR/face_detection_yunet_2023mar.onnx"
fetch "$SSD_PROTOTXT_URL" "$MODELS_DIR/deploy.prototxt"
fetch "$SSD_WEIGHTS_URL" "$MODELS_DIR/res10_300x300_ssd_iter_140000.caffemodel"

echo "done. Set video_analysis.models_dir = \"$MODELS_DIR\" in your ZeroClaw config."
