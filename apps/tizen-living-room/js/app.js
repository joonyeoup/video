/* app.js — one-screen state machine: idle → loading → result | error. */
(function () {
    "use strict";

    var CFG = window.APP_CONFIG;

    var KEY_ENTER = 13;
    var KEY_BACK = 10009; // Samsung remote Back/Return

    var LOADING_PHASES = [
        "Contacting agent…",
        "Extracting frames…",
        "Identifying events…",
        "Analyzing living room video…"
    ];

    var states = {
        idle: document.getElementById("state-idle"),
        loading: document.getElementById("state-loading"),
        result: document.getElementById("state-result"),
        error: document.getElementById("state-error")
    };
    var els = {
        idleVideoName: document.getElementById("idle-video-name"),
        btnAnalyze: document.getElementById("btn-analyze"),
        loadingStatus: document.getElementById("loading-status"),
        loadingElapsed: document.getElementById("loading-elapsed"),
        resultVideoName: document.getElementById("result-video-name"),
        eventList: document.getElementById("event-list"),
        resultMeta: document.getElementById("result-meta"),
        btnAgain: document.getElementById("btn-again"),
        errorTitle: document.getElementById("error-title"),
        errorDetail: document.getElementById("error-detail"),
        btnRetry: document.getElementById("btn-retry")
    };

    var current = null;
    var elapsedTimer = null;
    var phaseTimer = null;
    var requestSeq = 0; // ignore responses from superseded requests

    function videoName(path) {
        if (typeof path !== "string" || path === "") {
            return "";
        }
        var parts = path.split(/[\\/]/);
        return parts[parts.length - 1] || path;
    }

    function show(name) {
        current = name;
        Object.keys(states).forEach(function (key) {
            states[key].hidden = key !== name;
        });
        // Exactly one focusable element per state — focus it so Enter works.
        var focusTarget = {
            idle: els.btnAnalyze,
            result: els.btnAgain,
            error: els.btnRetry
        }[name];
        if (focusTarget) {
            focusTarget.focus();
        }
    }

    function stopTimers() {
        if (elapsedTimer !== null) {
            clearInterval(elapsedTimer);
            elapsedTimer = null;
        }
        if (phaseTimer !== null) {
            clearInterval(phaseTimer);
            phaseTimer = null;
        }
    }

    function startLoadingTimers() {
        var startedAt = Date.now();
        els.loadingElapsed.textContent = "0:00";
        els.loadingStatus.textContent = LOADING_PHASES[0];
        var phase = 0;

        elapsedTimer = setInterval(function () {
            var s = Math.floor((Date.now() - startedAt) / 1000);
            var mins = Math.floor(s / 60);
            var secs = s % 60;
            els.loadingElapsed.textContent =
                mins + ":" + (secs < 10 ? "0" : "") + secs;
        }, 1000);

        // Cosmetic only — the sync webhook gives no real progress. Hold the
        // final phase once reached.
        phaseTimer = setInterval(function () {
            if (phase < LOADING_PHASES.length - 1) {
                phase += 1;
                els.loadingStatus.textContent = LOADING_PHASES[phase];
            }
        }, 6000);
    }

    /* Merge plausible_events across timestamps, dedupe case-insensitively,
       preserve first-seen order and first-seen casing. */
    function collectEvents(result) {
        var seen = {};
        var events = [];
        (result.results || []).forEach(function (entry) {
            (entry.plausible_events || []).forEach(function (ev) {
                if (typeof ev !== "string") {
                    return;
                }
                var text = ev.trim();
                if (text === "") {
                    return;
                }
                var key = text.toLowerCase();
                if (!seen[key]) {
                    seen[key] = true;
                    events.push(text);
                }
            });
        });
        return events;
    }

    function renderResult(result) {
        els.resultVideoName.textContent =
            videoName(result.video_path) || videoName(CFG.VIDEO_PATH);

        var events = collectEvents(result);
        els.eventList.innerHTML = "";
        els.eventList.classList.toggle("dense", events.length > 7);
        if (events.length === 0) {
            events = ["No notable events detected in this video."];
        }
        events.forEach(function (text) {
            var li = document.createElement("li");
            li.textContent = text;
            els.eventList.appendChild(li);
        });

        var meta = result.metadata || {};
        var parts = [];
        if (meta.total_frames != null) {
            parts.push(meta.total_frames + " frames processed");
        }
        if (meta.frames_with_faces != null) {
            parts.push(meta.frames_with_faces + " with faces");
        }
        if (meta.processing_time) {
            parts.push(meta.processing_time);
        }
        els.resultMeta.textContent = parts.join(" · ");
        els.resultMeta.hidden = parts.length === 0;

        show("result");
    }

    function renderError(err) {
        els.errorTitle.textContent =
            (err && err.message) || "Couldn’t reach the analysis server";
        els.errorDetail.textContent = (err && err.detail) || "";
        show("error");
    }

    function analyze() {
        var seq = ++requestSeq;
        show("loading");
        startLoadingTimers();
        window.API.triggerAnalysis().then(function (result) {
            if (seq !== requestSeq) {
                return;
            }
            stopTimers();
            renderResult(result);
        }).catch(function (err) {
            if (seq !== requestSeq) {
                return;
            }
            stopTimers();
            renderError(err);
        });
    }

    function exitApp() {
        // Guarded so the app also runs in desktop Chrome.
        if (typeof tizen !== "undefined" &&
            tizen.application &&
            tizen.application.getCurrentApplication) {
            tizen.application.getCurrentApplication().exit();
        }
    }

    document.addEventListener("keydown", function (e) {
        if (e.keyCode === KEY_BACK) {
            e.preventDefault();
            exitApp();
            return;
        }
        if (e.keyCode === KEY_ENTER && current === "loading") {
            // Nothing to activate while loading; swallow the key.
            e.preventDefault();
        }
        // In idle/result/error the single focused <button> receives Enter
        // natively — no extra handling needed.
    });

    els.btnAnalyze.addEventListener("click", analyze);
    els.btnAgain.addEventListener("click", analyze);
    els.btnRetry.addEventListener("click", analyze);

    els.idleVideoName.textContent = videoName(CFG.VIDEO_PATH);
    show("idle");
})();
