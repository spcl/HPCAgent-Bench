"""Post to /v1/messages the way the agent driver does, and print what comes back.

610139 logged 87 x "500 Internal Server Error" from this endpoint and nothing else: the access log
carries the status line, never the body. Without the body we have been reasoning about which layer
rewrites the effort level from the outside. This asks the server directly, across the shapes the
driver can produce, and prints the failure verbatim.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

# What Claude Code sends at each level, plus the two the Qwen3.8 template accepts, so a failure can
# be attributed to the mapping rather than to the request being malformed.
SHAPES = (
    # The adapter sets chat_request.reasoning_effort ONLY when output_config.effort is present
    # (anthropic/serving.py:624-632), collapsing xhigh to max because the OpenAI Literal has no
    # xhigh. So a request WITHOUT output_config leaves the field unset and lets the server's
    # --default-chat-template-kwargs supply the level instead. These shapes separate the two.
    ("bare", {}),
    ("thinking-32768", {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 32768
        }
    }),
    ("output_config-xhigh", {
        "output_config": {
            "effort": "xhigh"
        }
    }),
    ("output_config-high", {
        "output_config": {
            "effort": "high"
        }
    }),
    ("output_config-medium", {
        "output_config": {
            "effort": "medium"
        }
    }),
    ("thinking+no-output-config", {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 8192
        }
    }),
)


def post(base: str, model: str, name: str, extra: dict, timeout: int) -> None:
    body = dict(extra)
    body.update({
        "model": model,
        "max_tokens": 64,
        "messages": [{
            "role": "user",
            "content": "Reply with the single word OK."
        }],
    })
    req = urllib.request.Request(f"{base}/v1/messages?beta=true",
                                 data=json.dumps(body).encode(),
                                 headers={
                                     "Content-Type": "application/json",
                                     "anthropic-version": "2023-06-01"
                                 })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
        blocks = payload.get("content") or []
        text = " ".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        print(f"  {name:16s} {resp.status} OK  {text.strip()[:80]!r}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"  {name:16s} {exc.code} {exc.reason}")
        for line in detail.splitlines()[:8]:
            print(f"      {line[:200]}")
    except Exception as exc:  # noqa: BLE001 -- transport failures are results here too
        print(f"  {name:16s} ERROR {type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    for name, extra in SHAPES:
        post(args.base, args.model, name, extra, args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
