//! `analyze_video` — privacy-preserving video analysis pipeline.
//!
//! Stages: (1) sample frames from a local video, (2) detect faces locally via
//! an embedded OpenCV Python helper (YuNet preferred, ResNet-SSD fallback) and
//! crop the face regions with padding, (3) send ONLY the face crops to the
//! configured OpenAI-compatible vision-language model for per-face estimates
//! and plausible-event generation, (4) aggregate every per-timestamp result
//! into one JSON report under `video_analysis.output_dir`.
//!
//! Full frames never leave the machine; the external VLM sees face crops only.
//! Configuration lives in the `[video_analysis]` section (see
//! `VideoAnalysisConfig`); the tool is registered only when
//! `video_analysis.enabled = true`.

use async_trait::async_trait;
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use zeroclaw_api::tool::{Tool, ToolOutput, ToolResult};
use zeroclaw_config::policy::SecurityPolicy;
use zeroclaw_config::schema::{FaceDetectorKind, VideoAnalysisConfig};

/// OpenCV frame-sampling + face-detection helper, written to a temp dir and
/// executed with `video_analysis.python_bin` on every run.
const FACE_DETECT_HELPER: &str = include_str!("video_analysis_helper.py");

/// Minimum detector confidence for a face crop to be kept.
const MIN_FACE_CONFIDENCE: f64 = 0.6;

/// System prompt pinning the VLM to the strict per-timestamp JSON schema.
const VLM_SYSTEM_PROMPT: &str = "You are a visual analysis assistant. You receive cropped face images taken at one \
timestamp from a living-room camera. For each face, estimate the person's age range, \
race/ethnicity, and emotion, and note how confident the crops allow you to be. Then list \
plausible events happening in the living room given those faces. These are best-effort \
visual estimates, not identifications. Respond ONLY with valid JSON matching exactly this \
schema, with no markdown fences and no prose outside the JSON:\n\
{\n\
  \"timestamp\": \"<video timestamp>\",\n\
  \"faces\": [\n\
    {\"face_id\": 1, \"age_estimate\": \"...\", \"race_estimate\": \"...\", \
\"emotion\": \"...\", \"confidence_notes\": \"...\"}\n\
  ],\n\
  \"plausible_events\": [\"...\", \"...\"]\n\
}";

/// Reminder appended when the first VLM reply is not parseable JSON.
const VLM_JSON_RETRY_REMINDER: &str = "Your previous reply was not valid JSON. Respond ONLY with valid JSON matching the \
schema you were given — no markdown fences, no explanations, nothing outside the JSON \
object.";

/// Manifest printed to stdout by the Python helper.
#[derive(Debug, Deserialize)]
struct DetectionManifest {
    detector_used: String,
    #[serde(default)]
    detector_note: Option<String>,
    frames_sampled: u64,
    frames_with_faces: u64,
    samples: Vec<DetectionSample>,
}

/// One sampled frame that contained at least one face.
#[derive(Debug, Deserialize)]
struct DetectionSample {
    timestamp: String,
    faces: Vec<DetectedFace>,
}

/// One face crop within a sample.
#[derive(Debug, Deserialize)]
struct DetectedFace {
    face_id: u32,
    crop_path: String,
}

/// Tool that runs the frame → face-crop → VLM → aggregate pipeline.
pub struct VideoAnalysisTool {
    security: Arc<SecurityPolicy>,
    config: VideoAnalysisConfig,
    client: reqwest::Client,
}

impl VideoAnalysisTool {
    pub fn new(security: Arc<SecurityPolicy>, config: VideoAnalysisConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.vlm.timeout_secs.max(1)))
            .build()
            .unwrap_or_default();
        Self {
            security,
            config,
            client,
        }
    }

    fn chat_completions_url(base_url: &str) -> String {
        format!("{}/v1/chat/completions", base_url.trim_end_matches('/'))
    }

    /// Extract a JSON object from VLM reply text: tolerate markdown fences and
    /// surrounding prose by falling back to the outermost `{ … }` span.
    fn parse_json_reply(content: &str) -> Option<Value> {
        let trimmed = content.trim();
        if let Ok(value) = serde_json::from_str::<Value>(trimmed)
            && value.is_object()
        {
            return Some(value);
        }
        let start = trimmed.find('{')?;
        let end = trimmed.rfind('}')?;
        if end <= start {
            return None;
        }
        serde_json::from_str::<Value>(&trimmed[start..=end])
            .ok()
            .filter(Value::is_object)
    }

    /// Build the multimodal user-message content for one timestamp: a text
    /// prompt followed by every face crop as a base64 data URL.
    fn build_user_content(sample: &DetectionSample, crops_b64: &[(u32, String)]) -> Value {
        let mut content = vec![json!({
            "type": "text",
            "text": format!(
                "Video timestamp {}: {} cropped face(s) follow, in face_id order ({}). \
                 Analyze them per the schema, using this exact timestamp value.",
                sample.timestamp,
                crops_b64.len(),
                crops_b64
                    .iter()
                    .map(|(id, _)| format!("face_id {id}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
        })];
        for (_, b64) in crops_b64 {
            content.push(json!({
                "type": "image_url",
                "image_url": { "url": format!("data:image/jpeg;base64,{b64}") },
            }));
        }
        Value::Array(content)
    }

    async fn post_chat(&self, url: &str, messages: &[Value]) -> Result<String, VlmError> {
        let body = json!({
            "model": self.config.vlm.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
        });
        let mut request = self.client.post(url).json(&body);
        if let Some(key) = self.config.vlm.api_key.as_deref() {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.map_err(|e| {
            if e.is_connect() || e.is_timeout() {
                VlmError::Unreachable(format!("{e}"))
            } else {
                VlmError::Request(format!("{e}"))
            }
        })?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| VlmError::Request(format!("failed to read response body: {e}")))?;
        if !status.is_success() {
            return Err(VlmError::Request(format!(
                "VLM server returned {status}: {}",
                text.chars().take(300).collect::<String>()
            )));
        }
        let parsed: Value = serde_json::from_str(&text)
            .map_err(|e| VlmError::Request(format!("non-JSON completion response: {e}")))?;
        parsed["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| {
                VlmError::Request("completion response missing choices[0].message.content".into())
            })
    }

    /// Analyze one timestamp's face crops. Malformed JSON from the model is
    /// retried once with an explicit valid-JSON reminder.
    async fn analyze_sample(
        &self,
        url: &str,
        sample: &DetectionSample,
        crops_b64: &[(u32, String)],
    ) -> Result<Value, VlmError> {
        let mut messages = vec![
            json!({ "role": "system", "content": VLM_SYSTEM_PROMPT }),
            json!({
                "role": "user",
                "content": Self::build_user_content(sample, crops_b64),
            }),
        ];
        let first_reply = self.post_chat(url, &messages).await?;
        if let Some(value) = Self::parse_json_reply(&first_reply) {
            return Ok(value);
        }

        messages.push(json!({ "role": "assistant", "content": first_reply }));
        messages.push(json!({ "role": "user", "content": VLM_JSON_RETRY_REMINDER }));
        let second_reply = self.post_chat(url, &messages).await?;
        Self::parse_json_reply(&second_reply).ok_or_else(|| {
            VlmError::MalformedJson(second_reply.chars().take(300).collect::<String>())
        })
    }

    /// Run the embedded Python helper; returns the parsed manifest or a
    /// user-facing error string.
    async fn run_face_detection(
        &self,
        video_path: &Path,
        sample_fps: f64,
        work_dir: &Path,
    ) -> Result<DetectionManifest, String> {
        let helper_path = work_dir.join("video_analysis_helper.py");
        let crops_dir = work_dir.join("crops");
        tokio::fs::write(&helper_path, FACE_DETECT_HELPER)
            .await
            .map_err(|e| format!("failed to stage detection helper: {e}"))?;
        tokio::fs::create_dir_all(&crops_dir)
            .await
            .map_err(|e| format!("failed to create crop directory: {e}"))?;

        let detector = match self.config.face_detector {
            FaceDetectorKind::Yunet => "yunet",
            FaceDetectorKind::ResnetSsd => "resnet_ssd",
        };
        let models_dir = self.security.resolve_tool_path(&self.config.models_dir);

        let output = tokio::process::Command::new(&self.config.python_bin)
            .arg(&helper_path)
            .arg("--video")
            .arg(video_path)
            .arg("--fps")
            .arg(sample_fps.to_string())
            .arg("--detector")
            .arg(detector)
            .arg("--models-dir")
            .arg(&models_dir)
            .arg("--out-dir")
            .arg(&crops_dir)
            .arg("--min-confidence")
            .arg(MIN_FACE_CONFIDENCE.to_string())
            .output()
            .await
            .map_err(|e| {
                format!(
                    "failed to launch face-detection helper via '{}': {e}. \
                     Set video_analysis.python_bin to a Python 3 interpreter with \
                     opencv-python-headless installed.",
                    self.config.python_bin
                )
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "face detection failed ({}): {}",
                output.status,
                stderr.trim()
            ));
        }
        serde_json::from_slice::<DetectionManifest>(&output.stdout)
            .map_err(|e| format!("face-detection helper produced invalid manifest: {e}"))
    }

    /// Resolve the report path inside the configured output directory.
    fn output_file_path(&self, video_path: &Path) -> PathBuf {
        let stem = video_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "video".to_string());
        self.security
            .resolve_tool_path(&self.config.output_dir)
            .join(format!("{stem}_analysis.json"))
    }
}

/// VLM-stage failure classification: unreachable servers abort the run,
/// per-request failures are recorded per timestamp and the run continues.
#[derive(Debug)]
enum VlmError {
    /// Transport-level failure (connect/timeout) — the server is unreachable.
    Unreachable(String),
    /// HTTP or protocol failure for one request.
    Request(String),
    /// The model never produced parseable JSON, even after the retry.
    MalformedJson(String),
}

impl std::fmt::Display for VlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable(e) => write!(f, "VLM server unreachable: {e}"),
            Self::Request(e) => write!(f, "VLM request failed: {e}"),
            Self::MalformedJson(reply) => {
                write!(f, "VLM returned malformed JSON after retry: {reply}")
            }
        }
    }
}

#[async_trait]
impl Tool for VideoAnalysisTool {
    fn name(&self) -> &str {
        "analyze_video"
    }

    fn description(&self) -> &str {
        "Analyze a video file for people and activity. Call this whenever the user asks to \
         analyze, inspect, or describe a video. Samples frames at a configurable rate, detects \
         faces locally (only cropped face regions are ever sent to the vision model), asks the \
         configured vision-language model for per-face age/ethnicity/emotion estimates plus \
         plausible living-room events, and writes an aggregated JSON report to the configured \
         output directory."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the video file (absolute or relative to workspace)"
                },
                "sample_fps": {
                    "type": "number",
                    "description": "Optional frame sampling rate override in frames per second \
                                    (default: video_analysis.sample_fps from config)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::Error::msg("Missing 'path' parameter"))?;
        let sample_fps = match args.get("sample_fps").and_then(Value::as_f64) {
            Some(fps) if fps > 0.0 => fps,
            Some(fps) => {
                return Ok(ToolResult::err(format!(
                    "sample_fps must be > 0, got {fps}"
                )));
            }
            None => self.config.sample_fps,
        };
        if sample_fps <= 0.0 {
            return Ok(ToolResult::err(
                "video_analysis.sample_fps must be > 0".to_string(),
            ));
        }
        if self.config.vlm.base_url.trim().is_empty() || self.config.vlm.model.trim().is_empty() {
            return Ok(ToolResult::err(
                "video_analysis.vlm.base_url and video_analysis.vlm.model must be set in config \
                 (the OpenAI-compatible VLM server to send face crops to)."
                    .to_string(),
            ));
        }

        // Path-allowlist checks run in the PathGuardedTool wrapper at
        // registration time; the read-side post-canonicalization boundary is
        // enforced here (mirrors image_info).
        let full_path = self.security.resolve_tool_path(path_str);
        let video_path = match tokio::fs::canonicalize(&full_path).await {
            Ok(path) => path,
            Err(e) => {
                let _ = self.security.record_action();
                let error = if e.kind() == std::io::ErrorKind::NotFound {
                    format!("Video not found: {path_str}")
                } else {
                    format!("Failed to resolve video path: {e}")
                };
                return Ok(ToolResult::err(error));
            }
        };
        if !self.security.is_resolved_path_readable(&video_path) {
            return Ok(ToolResult::err(
                "Resolved video path is outside the allowed readable roots.".to_string(),
            ));
        }

        let started = Instant::now();
        let work_dir = match tempfile::tempdir() {
            Ok(dir) => dir,
            Err(e) => {
                return Ok(ToolResult::err(format!(
                    "failed to create temporary work directory: {e}"
                )));
            }
        };

        // Stage 1+2: frame sampling and local face detection.
        let manifest = match self
            .run_face_detection(&video_path, sample_fps, work_dir.path())
            .await
        {
            Ok(manifest) => manifest,
            Err(e) => return Ok(ToolResult::err(e)),
        };

        // Stage 3: one VLM request per timestamp that contained faces.
        let url = Self::chat_completions_url(&self.config.vlm.base_url);
        let mut results: Vec<Value> = Vec::with_capacity(manifest.samples.len());
        let mut vlm_errors: u64 = 0;
        let mut fatal_error: Option<String> = None;
        for sample in &manifest.samples {
            let mut crops_b64: Vec<(u32, String)> = Vec::with_capacity(sample.faces.len());
            for face in &sample.faces {
                match tokio::fs::read(&face.crop_path).await {
                    Ok(bytes) => crops_b64.push((
                        face.face_id,
                        base64::engine::general_purpose::STANDARD.encode(bytes),
                    )),
                    Err(e) => {
                        vlm_errors += 1;
                        results.push(json!({
                            "timestamp": sample.timestamp,
                            "error": format!("failed to read face crop: {e}"),
                        }));
                    }
                }
            }
            if crops_b64.is_empty() {
                continue;
            }
            match self.analyze_sample(&url, sample, &crops_b64).await {
                Ok(mut value) => {
                    // Pin the timestamp to the detector's value so results stay
                    // aligned even when the model echoes it imprecisely.
                    if let Some(obj) = value.as_object_mut() {
                        obj.insert("timestamp".into(), json!(sample.timestamp));
                    }
                    results.push(value);
                }
                Err(VlmError::Unreachable(e)) => {
                    fatal_error = Some(format!(
                        "VLM server at {} unreachable: {e}. Check video_analysis.vlm.base_url \
                         and that the server is running.",
                        self.config.vlm.base_url
                    ));
                    break;
                }
                Err(e) => {
                    vlm_errors += 1;
                    results.push(json!({
                        "timestamp": sample.timestamp,
                        "error": e.to_string(),
                    }));
                }
            }
        }
        if let Some(error) = fatal_error {
            return Ok(ToolResult::err(error));
        }

        // Stage 4: aggregate and persist the report.
        let processing_time_secs = started.elapsed().as_secs_f64();
        let no_faces = manifest.frames_with_faces == 0;
        let report = json!({
            "video_path": video_path.display().to_string(),
            "generated_at": chrono::Utc::now().to_rfc3339(),
            "vlm_model": self.config.vlm.model,
            "detector_used": manifest.detector_used,
            "detector_note": manifest.detector_note,
            "sample_fps": sample_fps,
            "metadata": {
                "total_frames_processed": manifest.frames_sampled,
                "frames_with_faces": manifest.frames_with_faces,
                "vlm_request_errors": vlm_errors,
                "processing_time_secs": processing_time_secs,
            },
            "note": if no_faces { Some("no faces found in sampled frames") } else { None },
            "results": results,
        });

        let output_file = self.output_file_path(&video_path);
        if let Some(parent) = output_file.parent()
            && let Err(e) = tokio::fs::create_dir_all(parent).await
        {
            return Ok(ToolResult::err(format!(
                "failed to create output directory {}: {e}",
                parent.display()
            )));
        }
        let rendered = serde_json::to_string_pretty(&report).unwrap_or_else(|_| report.to_string());
        if let Err(e) = tokio::fs::write(&output_file, rendered).await {
            return Ok(ToolResult::err(format!(
                "failed to write analysis report {}: {e}",
                output_file.display()
            )));
        }

        let summary = json!({
            "output_file": output_file.display().to_string(),
            "detector_used": manifest.detector_used,
            "total_frames_processed": manifest.frames_sampled,
            "frames_with_faces": manifest.frames_with_faces,
            "timestamps_analyzed": results.len(),
            "vlm_request_errors": vlm_errors,
            "processing_time_secs": processing_time_secs,
        });
        let text = if no_faces {
            format!(
                "No faces found in {} sampled frames. Report written to {}",
                manifest.frames_sampled,
                output_file.display()
            )
        } else {
            format!(
                "Analyzed {} timestamp(s) with faces across {} sampled frames \
                 ({} VLM error(s)). Report written to {}",
                results.len(),
                manifest.frames_sampled,
                vlm_errors,
                output_file.display()
            )
        };
        Ok(ToolResult::ok(ToolOutput::json_with_text(summary, text)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    use zeroclaw_config::autonomy::AutonomyLevel;

    fn test_security() -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: std::env::temp_dir(),
            workspace_only: false,
            forbidden_paths: vec![],
            ..SecurityPolicy::default()
        })
    }

    fn test_config(base_url: &str) -> VideoAnalysisConfig {
        VideoAnalysisConfig {
            enabled: true,
            vlm: zeroclaw_config::schema::VideoAnalysisVlmConfig {
                base_url: base_url.to_string(),
                model: "test-vlm".to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn make_tool(base_url: &str) -> VideoAnalysisTool {
        VideoAnalysisTool::new(test_security(), test_config(base_url))
    }

    fn sample_with_one_face(crop_path: &str) -> DetectionSample {
        DetectionSample {
            timestamp: "00:00:01.000".to_string(),
            faces: vec![DetectedFace {
                face_id: 1,
                crop_path: crop_path.to_string(),
            }],
        }
    }

    fn completion_body(content: &str) -> Value {
        json!({
            "choices": [ { "message": { "role": "assistant", "content": content } } ]
        })
    }

    #[test]
    fn tool_name_and_spec() {
        let tool = make_tool("http://localhost:9");
        assert_eq!(tool.name(), "analyze_video");
        assert!(!tool.description().is_empty());
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["path"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .is_some_and(|r| r.contains(&json!("path")))
        );
    }

    #[test]
    fn chat_url_strips_trailing_slash() {
        assert_eq!(
            VideoAnalysisTool::chat_completions_url("http://10.0.0.2:8000/"),
            "http://10.0.0.2:8000/v1/chat/completions"
        );
        assert_eq!(
            VideoAnalysisTool::chat_completions_url("http://10.0.0.2:8000"),
            "http://10.0.0.2:8000/v1/chat/completions"
        );
    }

    #[test]
    fn parse_json_reply_accepts_plain_object() {
        let value = VideoAnalysisTool::parse_json_reply(r#"{"timestamp":"t","faces":[]}"#).unwrap();
        assert_eq!(value["timestamp"], "t");
    }

    #[test]
    fn parse_json_reply_strips_fences_and_prose() {
        let reply = "Here you go:\n```json\n{\"faces\": [], \"plausible_events\": []}\n```";
        let value = VideoAnalysisTool::parse_json_reply(reply).unwrap();
        assert!(value["faces"].as_array().is_some());
    }

    #[test]
    fn parse_json_reply_rejects_garbage() {
        assert!(VideoAnalysisTool::parse_json_reply("not json at all").is_none());
        assert!(VideoAnalysisTool::parse_json_reply("[1, 2, 3]").is_none());
    }

    #[tokio::test]
    async fn execute_requires_path() {
        let tool = make_tool("http://localhost:9");
        assert!(tool.execute(json!({})).await.is_err());
    }

    #[tokio::test]
    async fn execute_rejects_unconfigured_vlm() {
        let tool = VideoAnalysisTool::new(test_security(), VideoAnalysisConfig::default());
        let result = tool.execute(json!({"path": "video.mp4"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("vlm.base_url"));
    }

    #[tokio::test]
    async fn execute_rejects_missing_video() {
        let tool = make_tool("http://localhost:9");
        let result = tool
            .execute(json!({"path": "/tmp/zeroclaw_no_such_video.mp4"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("not found"));
    }

    #[tokio::test]
    async fn execute_rejects_nonpositive_sample_fps() {
        let tool = make_tool("http://localhost:9");
        let result = tool
            .execute(json!({"path": "video.mp4", "sample_fps": 0.0}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("sample_fps"));
    }

    #[tokio::test]
    async fn analyze_sample_parses_first_reply() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_body(
                r#"{"timestamp":"x","faces":[],"plausible_events":[]}"#,
            )))
            .expect(1)
            .mount(&server)
            .await;

        let tool = make_tool(&server.uri());
        let url = VideoAnalysisTool::chat_completions_url(&server.uri());
        let sample = sample_with_one_face("unused.jpg");
        let value = tool
            .analyze_sample(&url, &sample, &[(1, "aGk=".to_string())])
            .await
            .unwrap();
        assert!(value["faces"].as_array().is_some());
    }

    #[tokio::test]
    async fn analyze_sample_retries_malformed_json_once() {
        let server = MockServer::start().await;
        // First call: prose. Second call (retry with reminder): valid JSON.
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(completion_body("Sure! The people look happy.")),
            )
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_body(
                r#"{"timestamp":"x","faces":[{"face_id":1}],"plausible_events":["tv"]}"#,
            )))
            .expect(1)
            .mount(&server)
            .await;

        let tool = make_tool(&server.uri());
        let url = VideoAnalysisTool::chat_completions_url(&server.uri());
        let sample = sample_with_one_face("unused.jpg");
        let value = tool
            .analyze_sample(&url, &sample, &[(1, "aGk=".to_string())])
            .await
            .unwrap();
        assert_eq!(value["plausible_events"][0], "tv");
    }

    #[tokio::test]
    async fn analyze_sample_reports_malformed_after_retry() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(completion_body("still not json")),
            )
            .expect(2)
            .mount(&server)
            .await;

        let tool = make_tool(&server.uri());
        let url = VideoAnalysisTool::chat_completions_url(&server.uri());
        let sample = sample_with_one_face("unused.jpg");
        let err = tool
            .analyze_sample(&url, &sample, &[(1, "aGk=".to_string())])
            .await
            .unwrap_err();
        assert!(matches!(err, VlmError::MalformedJson(_)));
    }

    #[tokio::test]
    async fn post_chat_classifies_unreachable_server() {
        // Port 9 (discard) is reliably closed; connect must fail fast.
        let tool = make_tool("http://127.0.0.1:9");
        let err = tool
            .post_chat(
                "http://127.0.0.1:9/v1/chat/completions",
                &[json!({"role": "user", "content": "hi"})],
            )
            .await
            .unwrap_err();
        assert!(matches!(err, VlmError::Unreachable(_)));
    }

    #[test]
    fn user_content_carries_one_image_per_crop() {
        let sample = sample_with_one_face("a.jpg");
        let content = VideoAnalysisTool::build_user_content(
            &sample,
            &[(1, "AAAA".to_string()), (2, "BBBB".to_string())],
        );
        let items = content.as_array().unwrap();
        assert_eq!(items.len(), 3); // text + 2 images
        assert_eq!(items[0]["type"], "text");
        assert!(
            items[1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/jpeg;base64,")
        );
    }
}
