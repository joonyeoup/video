/* Living Room Analysis — all knobs live here. */
window.APP_CONFIG = {
    // Demo without a backend: true returns MOCK_RESULT after MOCK_DELAY_MS.
    MOCK_MODE: false,

    // ZeroClaw gateway base URL (no trailing slash).
    SERVER_URL: "http://192.168.0.10:9000",

    // Video the agent should analyze. Baked into the trigger message below.
    VIDEO_PATH: "/data/videos/living_room_0712.mp4",

    // Message sent to POST /webhook. {video_path} is replaced with VIDEO_PATH.
    // Keep the "respond with only the raw JSON" instruction — the parser
    // tolerates prose around the JSON, but clean output is more reliable.
    MESSAGE_TEMPLATE:
        "Use the analyze_video tool on the video at {video_path} and respond " +
        "with only the raw JSON result, no commentary.",

    // Auth — leave empty if the gateway doesn't require them.
    BEARER_TOKEN: "",     // Authorization: Bearer <token> (gateway pairing)
    WEBHOOK_SECRET: "",   // X-Webhook-Secret header (optional extra layer)

    // Optional ?agent=<alias> on the webhook call. Empty = gateway default.
    AGENT_ALIAS: "",

    // The webhook is synchronous and video analysis is slow — allow plenty.
    REQUEST_TIMEOUT_MS: 10 * 60 * 1000,

    MOCK_DELAY_MS: 2500,
    MOCK_RESULT: {
        video_path: "/data/videos/living_room_0712.mp4",
        metadata: {
            total_frames: 128,
            frames_with_faces: 42,
            processing_time: "3m 12s"
        },
        results: [
            {
                timestamp: "00:00:04",
                plausible_events: [
                    "A person is watching television on the couch",
                    "Someone is holding a remote control"
                ]
            },
            {
                timestamp: "00:00:12",
                plausible_events: [
                    "a person is watching television on the couch",
                    "A child is playing on the floor near the coffee table"
                ]
            },
            {
                timestamp: "00:00:21",
                plausible_events: [
                    "Two people are having a conversation",
                    "Someone is drinking from a mug"
                ]
            }
        ]
    }
};
