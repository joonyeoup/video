# Video Analysis (`analyze_video` tool)

Privacy-preserving video analysis pipeline, built in as a native ZeroClaw tool:

1. **Frame sampling** — frames are sampled from the video at `sample_fps` (default 1 fps) using OpenCV, in memory.
2. **Local face detection** — each sampled frame runs through YuNet (preferred) or ResNet-SSD via a small embedded Python helper. Frames without faces are discarded. Face regions are cropped with ~20% padding. **Full frames never leave the machine** — only face crops go to the VLM.
3. **VLM analysis** — face crops from the same timestamp are batched into one request to your OpenAI-compatible vision server (e.g. vLLM serving Qwen3-VL). The model returns strict JSON with per-face age/race/emotion estimates and plausible living-room events; malformed replies are retried once with a valid-JSON reminder.
4. **Aggregation** — all per-timestamp results plus metadata (frames processed, frames with faces, processing time) are written to `<output_dir>/<video_name>_analysis.json`.

## Setup

```bash
# 1. Python helper dependencies (any Python 3.9+)
python3 -m pip install opencv-python-headless

# 2. Detector weights (YuNet ONNX + ResNet-SSD fallback)
scripts/download_face_models.sh ./models

# 3. Build ZeroClaw
cargo build --release
```

## Configuration

Add to your ZeroClaw config (`~/.zeroclaw/config.toml`) — replace the placeholders:

```toml
[video_analysis]
enabled = true
sample_fps = 1.0                    # frames sampled per second of video
output_dir = "./analysis_output"    # reports land here (relative = workspace)
face_detector = "yunet"             # or "resnet_ssd"
models_dir = "./models"             # where download_face_models.sh put weights
python_bin = "python3"              # interpreter with opencv installed

[video_analysis.vlm]
base_url = "http://<SERVER_IP>:<PORT>"   # <-- YOUR vLLM server (no path suffix)
model = "<MODEL_NAME>"                   # <-- e.g. "Qwen/Qwen3-VL-8B-Instruct"
# api_key = "..."                        # optional bearer token
timeout_secs = 120
```

The tool is only registered with the agent when `enabled = true`.

## Usage

Ask the agent in natural language:

```bash
zeroclaw agent --message "analyze the video at /path/to/video.mp4"
```

The LLM invokes `analyze_video`, and the reply includes the report path, e.g.
`./analysis_output/video_analysis.json`:

```json
{
  "video_path": "/path/to/video.mp4",
  "vlm_model": "<MODEL_NAME>",
  "detector_used": "yunet",
  "metadata": {
    "total_frames_processed": 120,
    "frames_with_faces": 34,
    "vlm_request_errors": 0,
    "processing_time_secs": 88.4
  },
  "results": [
    {
      "timestamp": "00:00:12.000",
      "faces": [
        {"face_id": 1, "age_estimate": "30-40", "race_estimate": "…",
         "emotion": "relaxed", "confidence_notes": "profile view, low light"}
      ],
      "plausible_events": ["watching TV together", "conversation on the couch"]
    }
  ]
}
```

Videos with zero detected faces still produce a report with
`"note": "no faces found in sampled frames"`.

## Error handling

- **VLM unreachable** — the run aborts with a message naming `vlm.base_url`.
- **Malformed VLM JSON** — retried once with a strict-JSON reminder; if still malformed, that timestamp is recorded as an `error` entry and the run continues.
- **Missing weights / Python / OpenCV** — the tool fails with the exact install command or download script to run.

## Smoke test

```bash
scripts/smoke_video_analysis.sh /path/to/sample.mp4 ./models
```

Verifies Python+OpenCV, model weights, and the frame-sampling/face-detection stage end-to-end (no VLM needed), then prints the agent command for the full run.

## Caveats

Age, race/ethnicity, and emotion values are best-effort visual guesses from a small VLM on cropped faces — treat them as noisy annotations, not ground truth, and use the output only for footage you have the right to analyze.
