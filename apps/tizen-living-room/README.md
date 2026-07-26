# Living Room Analysis — Tizen TV demo app

Minimal Samsung Smart TV (Tizen 7.0+) web app that triggers the ZeroClaw
video-analysis pipeline on a remote server and shows the plausible living-room
events on one polished result card. UI only — all analysis happens on the
ZeroClaw gateway.

```
config.xml        Tizen packaging manifest (app id, network access)
index.html        Single screen, four states (idle / loading / result / error)
css/style.css     1920x1080 fixed, 10-foot light theme
js/config.js      ← everything you'd want to change lives here
js/api.js         POST /webhook call + tolerant JSON extraction
js/app.js         State machine, remote-key handling, rendering
icon.png          Placeholder icon — replace with real artwork when convenient
```

## Configure

Edit `js/config.js`:

| Key | What it does |
|---|---|
| `MOCK_MODE` | `true` = demo the full flow with canned JSON, no backend needed |
| `SERVER_URL` | ZeroClaw gateway base URL, e.g. `http://192.168.0.10:9000` |
| `VIDEO_PATH` | Server-side path of the video the agent should analyze |
| `MESSAGE_TEMPLATE` | The natural-language trigger sent to `POST /webhook` |
| `BEARER_TOKEN` | Fill in if the gateway requires pairing (`Authorization: Bearer …`) |
| `WEBHOOK_SECRET` | Fill in if the gateway sets a webhook secret (`X-Webhook-Secret`) |
| `AGENT_ALIAS` | Optional `?agent=<alias>` to pick a configured agent |
| `REQUEST_TIMEOUT_MS` | Sync call — keep generous (default 10 min) |

The app calls the gateway's synchronous webhook:

```bash
curl -X POST http://<SERVER_IP>:<PORT>/webhook \
  -H 'Content-Type: application/json' \
  -d '{"message": "Use the analyze_video tool on the video at /data/videos/living_room_0712.mp4 and respond with only the raw JSON result, no commentary."}'
# → {"response": "<agent text containing the result JSON>", "model": "..."}
```

`api.js` extracts the result JSON even if the agent wraps it in prose or a
```` ```json ```` fence, and treats any reply without a `results` array as a
malformed-response error (friendly message + Retry).

## Iterate in desktop Chrome

No build step — just serve the folder (`fetch` needs http, not `file://`):

```bash
cd apps/tizen-living-room
python3 -m http.server 8080
# open http://localhost:8080 — set MOCK_MODE: true for the canned demo
```

All `tizen.*` calls are guarded, so everything except remote-key exit works
in a normal browser. Enter/click activates the focused button.

## Package with the Tizen CLI

Prereqs: Tizen Studio (or its CLI package) with the TV extension, and a
Samsung certificate profile (Tizen Studio → Certificate Manager, or
`tizen certificate` / `tizen security-profiles add`). Below, `<profile>` is
that certificate profile's name.

```bash
cd apps/tizen-living-room

# 1. Build the web app (outputs to .buildResult/)
tizen build-web -- .

# 2. Package as a signed .wgt
tizen package -t wgt -s <profile> -- .buildResult
# → .buildResult/LivingRoom.wgt (name comes from the <name> in config.xml)
```

## Install on the TV

1. On the TV: **Apps → 12345** (enter with remote) → turn **Developer mode**
   ON, set **Host PC IP** to your Mac's IP → restart the TV.
2. From the Mac:

```bash
sdb connect <TV_IP>:26101
sdb devices                          # confirm the TV shows up
tizen install -n .buildResult/LivingRoom.wgt -t <device serial from sdb devices>
```

3. The app appears in the TV's Apps panel. Launch it; **Enter** triggers
   Analyze / Retry, **Back** exits.

To reinstall after a change: rebuild, repackage, and run `tizen install`
again (same app id overwrites in place).

## Notes

- The TV must be able to reach `SERVER_URL` (same LAN / no client isolation).
  The app talks plain `http`, which `config.xml` permits via
  `<access origin="*">`; tighten that to the exact origin for anything
  beyond the demo lab.
- Per-face data in the backend JSON is intentionally ignored — the card only
  surfaces merged, deduplicated `plausible_events` plus the metadata footer.
