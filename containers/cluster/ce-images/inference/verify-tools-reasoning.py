"""Prove a served endpoint actually round-trips tool calls and reasoning content.

The concurrency sweep in the smoke scripts SENDS tools, but it only counts tokens: a server whose
tool-call parser is missing or misnamed returns prose describing the call and the sweep scores it
as a healthy response. SGLang in particular starts fine with an unnamed or wrong --tool-call-parser
and fails at the FIRST request, which reads as a serving bug rather than a launch one, so the
failure has to be asserted rather than eyeballed.

Reasoning is the same shape: an engine that drops reasoning_effort still answers, just without the
thinking budget the arm was configured for, and nothing in a tok/s number shows it.

Exit code is 0 only when every requested check passed.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


def post_chat(base: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def check_tool_call(base: str, model: str, timeout: int) -> tuple[bool, str]:
    """A named tool must come back as a structured tool_calls entry, not as prose about it."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Zurich? Use the tool."}],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 512,
    }
    try:
        data = post_chat(base, body, timeout)
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, f"request failed: {exc}"

    message = data["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if not calls:
        content = (message.get("content") or "")[:200]
        return False, f"no tool_calls in response; content={content!r}"

    fn = calls[0]["function"]
    if fn["name"] != "get_weather":
        return False, f"wrong tool name: {fn['name']!r}"
    try:
        args = json.loads(fn["arguments"])
    except json.JSONDecodeError as exc:
        return False, f"arguments are not valid JSON: {exc}; raw={fn['arguments']!r}"
    if "city" not in args:
        return False, f"arguments missing 'city': {args!r}"
    return True, f"tool_calls[0]={fn['name']}({args})"


def check_reasoning(base: str, model: str, effort: str, timeout: int) -> tuple[bool, str]:
    """Reasoning must surface as reasoning_content, and the requested effort must be accepted."""
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "A farmer has 12 sheep, buys 3 more, then sells 5. How many remain? Think it through.",
            }
        ],
        "temperature": 0,
        "max_tokens": 2048,
    }
    if effort:
        body["reasoning_effort"] = effort
    try:
        data = post_chat(base, body, timeout)
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, f"request failed (effort={effort!r}): {exc}"

    message = data["choices"][0]["message"]
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    if not reasoning:
        return False, f"no reasoning_content returned (effort={effort!r}); content={content[:200]!r}"
    if "10" not in content:
        return False, f"wrong answer; expected 10, content={content[:200]!r}"
    return True, f"reasoning_content={len(reasoning)} chars, answer ok (effort={effort!r})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. http://nid002664:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning-effort", default="max", help="empty string to omit the field")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--skip-tools", action="store_true")
    ap.add_argument("--skip-reasoning", action="store_true")
    args = ap.parse_args()

    results: list[tuple[str, bool, str]] = []
    if not args.skip_tools:
        ok, detail = check_tool_call(args.base, args.model, args.timeout)
        results.append(("tool-call", ok, detail))
    if not args.skip_reasoning:
        ok, detail = check_reasoning(args.base, args.model, args.reasoning_effort, args.timeout)
        results.append(("reasoning", ok, detail))

    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)

    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("all endpoint checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
