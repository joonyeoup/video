#!/usr/bin/env python3
"""Frame-sampling and local face-detection helper for ZeroClaw's `analyze_video` tool.

Embedded into the Rust tool via include_str! and written to a temp dir per run.
Reads a video with OpenCV, samples frames at --fps, detects faces (YuNet
preferred, ResNet-SSD Caffe fallback), writes padded face crops as JPEGs to
--out-dir, and prints a JSON manifest to stdout. Full frames never leave this
process; only the face crops are persisted.

Requires: opencv-python or opencv-python-headless (>= 4.8 for FaceDetectorYN).
Model weights: see scripts/download_face_models.sh.
"""

import argparse
import json
import os
import sys

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SSD_PROTOTXT_FILENAME = "deploy.prototxt"
SSD_WEIGHTS_FILENAME = "res10_300x300_ssd_iter_140000.caffemodel"
CROP_PADDING = 0.20  # fraction of box width/height added on each side


def die(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def load_cv2():
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        die(
            "python module 'cv2' not found. Install it with: "
            "pip install opencv-python-headless"
        )
    return cv2


class YunetDetector:
    name = "yunet"

    def __init__(self, cv2, model_path: str, min_confidence: float):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("cv2.FaceDetectorYN unavailable (OpenCV >= 4.8 required)")
        self._cv2 = cv2
        self._det = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), min_confidence, 0.3, 5000
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame)
        results = []
        if faces is not None:
            for row in faces:
                x, y, bw, bh = (int(v) for v in row[:4])
                results.append((x, y, bw, bh, float(row[-1])))
        return results


class ResnetSsdDetector:
    name = "resnet_ssd"

    def __init__(self, cv2, prototxt_path: str, weights_path: str, min_confidence: float):
        for p in (prototxt_path, weights_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
        self._cv2 = cv2
        self._net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
        self._min_confidence = min_confidence

    def detect(self, frame):
        cv2 = self._cv2
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self._min_confidence:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            if x2 <= x1 or y2 <= y1:
                continue
            results.append((x1, y1, x2 - x1, y2 - y1, confidence))
        return results


def build_detector(cv2, kind: str, models_dir: str, min_confidence: float):
    """Return (detector, note). YuNet falls back to ResNet-SSD when unavailable."""
    yunet_path = os.path.join(models_dir, YUNET_FILENAME)
    ssd_prototxt = os.path.join(models_dir, SSD_PROTOTXT_FILENAME)
    ssd_weights = os.path.join(models_dir, SSD_WEIGHTS_FILENAME)

    if kind == "yunet":
        try:
            return YunetDetector(cv2, yunet_path, min_confidence), None
        except (FileNotFoundError, RuntimeError, cv2.error) as e:
            note = f"yunet unavailable ({e}); falling back to resnet_ssd"
            try:
                return ResnetSsdDetector(cv2, ssd_prototxt, ssd_weights, min_confidence), note
            except (FileNotFoundError, cv2.error):
                die(
                    f"no face detector available: {note}, and ResNet-SSD weights "
                    f"missing from {models_dir}. Run scripts/download_face_models.sh"
                )
    elif kind == "resnet_ssd":
        try:
            return ResnetSsdDetector(cv2, ssd_prototxt, ssd_weights, min_confidence), None
        except (FileNotFoundError, cv2.error) as e:
            die(
                f"resnet_ssd unavailable ({e}). Run scripts/download_face_models.sh "
                f"to place weights in {models_dir}"
            )
    else:
        die(f"unknown detector kind: {kind}")


def clamp_crop(frame, x, y, w, h):
    fh, fw = frame.shape[:2]
    pad_x = int(w * CROP_PADDING)
    pad_y = int(h * CROP_PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def format_timestamp(seconds: float) -> str:
    m, s = divmod(seconds, 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--fps", type=float, required=True, help="sample rate in frames/sec")
    parser.add_argument("--detector", choices=["yunet", "resnet_ssd"], required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--out-dir", required=True, help="directory for face-crop JPEGs")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    args = parser.parse_args()

    cv2 = load_cv2()

    if args.fps <= 0:
        die(f"--fps must be > 0, got {args.fps}")
    os.makedirs(args.out_dir, exist_ok=True)

    detector, detector_note = build_detector(
        cv2, args.detector, args.models_dir, args.min_confidence
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        die(f"could not open video: {args.video}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0:
        native_fps = 30.0  # container did not report fps; assume common default
    step = max(1, round(native_fps / args.fps))

    samples = []
    frames_sampled = 0
    frame_index = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % step != 0:
            continue
        frames_sampled += 1

        timestamp_sec = frame_index / native_fps
        faces = []
        for face_id, (x, y, w, h, confidence) in enumerate(detector.detect(frame), start=1):
            crop = clamp_crop(frame, x, y, w, h)
            if crop is None:
                continue
            crop_path = os.path.join(
                args.out_dir, f"t{timestamp_sec:09.3f}_face{face_id}.jpg"
            )
            if not cv2.imwrite(crop_path, crop):
                die(f"failed to write face crop: {crop_path}")
            faces.append(
                {
                    "face_id": face_id,
                    "crop_path": crop_path,
                    "bbox": [x, y, w, h],
                    "detection_confidence": round(confidence, 4),
                }
            )
        if faces:
            samples.append(
                {
                    "timestamp_sec": round(timestamp_sec, 3),
                    "timestamp": format_timestamp(timestamp_sec),
                    "frame_index": frame_index,
                    "faces": faces,
                }
            )
    cap.release()

    manifest = {
        "video": args.video,
        "native_fps": round(native_fps, 3),
        "detector_used": detector.name,
        "detector_note": detector_note,
        "frames_sampled": frames_sampled,
        "frames_with_faces": len(samples),
        "samples": samples,
    }
    json.dump(manifest, sys.stdout)


if __name__ == "__main__":
    main()
