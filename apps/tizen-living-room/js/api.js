/* api.js — talks to the ZeroClaw gateway (POST /webhook, synchronous).
 *
 * Success body from the gateway: {"response": "<agent text>", "model": "..."}
 * Error body:                    {"error": "..."}
 *
 * The agent's text reply *contains* the analysis JSON but may wrap it in
 * prose or a ```json fence, so extraction is deliberately tolerant.
 */
(function () {
    "use strict";

    var CFG = window.APP_CONFIG;

    function ApiError(message, detail) {
        this.name = "ApiError";
        this.message = message;
        this.detail = detail || "";
    }
    ApiError.prototype = Object.create(Error.prototype);

    /* Pull the analysis result object out of the agent's free-text reply. */
    function extractResultJson(text) {
        if (typeof text !== "string" || text.trim() === "") {
            throw new ApiError(
                "The server responded in an unexpected format.",
                "Empty response from agent."
            );
        }
        var candidates = [];
        var trimmed = text.trim();
        candidates.push(trimmed);

        // ```json ... ``` fences, if any.
        var fence = /```(?:json)?\s*([\s\S]*?)```/g;
        var m;
        while ((m = fence.exec(text)) !== null) {
            candidates.push(m[1].trim());
        }

        // Outermost brace span as a last resort.
        var first = text.indexOf("{");
        var last = text.lastIndexOf("}");
        if (first !== -1 && last > first) {
            candidates.push(text.slice(first, last + 1));
        }

        for (var i = 0; i < candidates.length; i++) {
            var parsed;
            try {
                parsed = JSON.parse(candidates[i]);
            } catch (e) {
                continue;
            }
            if (parsed && typeof parsed === "object" &&
                Object.prototype.toString.call(parsed.results) === "[object Array]") {
                return parsed;
            }
        }
        throw new ApiError(
            "The server responded in an unexpected format.",
            "No analysis JSON found in the agent reply."
        );
    }

    function mockAnalysis() {
        return new Promise(function (resolve) {
            setTimeout(function () {
                resolve(CFG.MOCK_RESULT);
            }, CFG.MOCK_DELAY_MS);
        });
    }

    function realAnalysis() {
        var url = CFG.SERVER_URL + "/webhook";
        if (CFG.AGENT_ALIAS) {
            url += "?agent=" + encodeURIComponent(CFG.AGENT_ALIAS);
        }
        var headers = { "Content-Type": "application/json" };
        if (CFG.BEARER_TOKEN) {
            headers["Authorization"] = "Bearer " + CFG.BEARER_TOKEN;
        }
        if (CFG.WEBHOOK_SECRET) {
            headers["X-Webhook-Secret"] = CFG.WEBHOOK_SECRET;
        }
        var message = CFG.MESSAGE_TEMPLATE.replace("{video_path}", CFG.VIDEO_PATH);

        var controller = typeof AbortController !== "undefined"
            ? new AbortController()
            : null;
        var timer = null;
        if (controller) {
            timer = setTimeout(function () {
                controller.abort();
            }, CFG.REQUEST_TIMEOUT_MS);
        }

        return fetch(url, {
            method: "POST",
            headers: headers,
            body: JSON.stringify({ message: message }),
            signal: controller ? controller.signal : undefined
        }).then(function (res) {
            return res.json().catch(function () {
                throw new ApiError(
                    "The server responded in an unexpected format.",
                    "HTTP " + res.status + " with a non-JSON body."
                );
            }).then(function (body) {
                if (!res.ok) {
                    throw new ApiError(
                        "The analysis request was rejected.",
                        body && body.error
                            ? body.error
                            : "HTTP " + res.status
                    );
                }
                return extractResultJson(body.response);
            });
        }).catch(function (err) {
            if (err instanceof ApiError) {
                throw err;
            }
            if (err && err.name === "AbortError") {
                throw new ApiError(
                    "The analysis timed out.",
                    "No response within " +
                        Math.round(CFG.REQUEST_TIMEOUT_MS / 60000) + " minutes."
                );
            }
            throw new ApiError(
                "Couldn’t reach the analysis server.",
                (err && err.message ? err.message + " — " : "") +
                    "Check that the gateway at " + CFG.SERVER_URL +
                    " is running and reachable from the TV."
            );
        }).finally(function () {
            if (timer !== null) {
                clearTimeout(timer);
            }
        });
    }

    window.API = {
        ApiError: ApiError,
        /* Returns a Promise resolving to the analysis result object. */
        triggerAnalysis: function () {
            return CFG.MOCK_MODE ? mockAnalysis() : realAnalysis();
        }
    };
})();
