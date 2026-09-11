# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

"""Provider-agnostic web search -- one call, any popular backend, keyed by env var.

A thin, stdlib-only client (``urllib``, no third-party dep) so an agent (or the
harness) can search the web through whichever provider the environment has a key
for. The provider is picked, in order, from:

1. an explicit :class:`WebSearchConfig` ``provider``;
2. ``$HPCAGENT_BENCH_WEBSEARCH_PROVIDER``;
3. auto-detect -- the first provider (in :class:`Provider` declaration order) whose
   API key(s) are present in the environment.

Every backend normalizes to the SAME :class:`SearchResponse` (a list of
:class:`SearchResult` ``{title, url, content}`` plus an optional ``answer``), so a
caller never branches on the provider. The HTTP transport is injectable
(``transport=``) so the loop is unit-testable with no network.

Supported providers and their env keys::

    tavily      TAVILY_API_KEY
    serper      SERPER_API_KEY
    brave       BRAVE_API_KEY | BRAVE_SEARCH_API_KEY
    exa         EXA_API_KEY
    google_cse  GOOGLE_CSE_API_KEY  (+ GOOGLE_CSE_ID, the engine cx)
    bing        BING_SEARCH_API_KEY | BING_SUBSCRIPTION_KEY
    serpapi     SERPAPI_API_KEY | SERPAPI_KEY
    you         YDC_API_KEY | YOU_API_KEY
    jina        JINA_API_KEY
    perplexity  PERPLEXITY_API_KEY   (an answer engine: fills ``answer`` + citations)

Adding a provider is one :class:`Provider` member, one ``_ENV_KEYS`` row, one
request builder, and one parser -- no caller change.

    python -m hpcagent_bench.websearch "fast gemm avx512" --max-results 5
    python -m hpcagent_bench.websearch --list          # which providers have a key here
"""

import argparse
import dataclasses
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, cast


class Provider(str, Enum):
    """A web-search backend. Declaration order is the auto-detect priority."""

    TAVILY = "tavily"
    SERPER = "serper"
    BRAVE = "brave"
    EXA = "exa"
    GOOGLE_CSE = "google_cse"
    BING = "bing"
    SERPAPI = "serpapi"
    YOU = "you"
    JINA = "jina"
    PERPLEXITY = "perplexity"


#: provider -> the env var(s) that hold its API key (any one present = configured).
_ENV_KEYS: dict[Provider, tuple[str, ...]] = {
    Provider.TAVILY: ("TAVILY_API_KEY",),
    Provider.SERPER: ("SERPER_API_KEY",),
    Provider.BRAVE: ("BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY"),
    Provider.EXA: ("EXA_API_KEY",),
    Provider.GOOGLE_CSE: ("GOOGLE_CSE_API_KEY",),
    Provider.BING: ("BING_SEARCH_API_KEY", "BING_SUBSCRIPTION_KEY"),
    Provider.SERPAPI: ("SERPAPI_API_KEY", "SERPAPI_KEY"),
    Provider.YOU: ("YDC_API_KEY", "YOU_API_KEY"),
    Provider.JINA: ("JINA_API_KEY",),
    Provider.PERPLEXITY: ("PERPLEXITY_API_KEY",),
}


class WebSearchError(RuntimeError):
    """A configuration or transport failure in a web-search call."""


@dataclass(frozen=True)
class WebSearchConfig:
    """How to search -- a config object, never a bag of positional strings.

    ``provider`` ``None`` auto-detects from the environment. ``api_key`` / ``cse_id``
    override the env-resolved credentials (e.g. to pass a key held elsewhere).
    """

    provider: Provider | None = None
    max_results: int = 5
    timeout: float = 30.0
    api_key: str | None = None
    cse_id: str | None = None  # google_cse only (the search-engine cx)

    def __post_init__(self) -> None:
        if self.provider is not None:
            object.__setattr__(self, "provider", Provider(self.provider))  # coerce/validate a string
        if int(self.max_results) < 1:
            raise ValueError(f"max_results must be >= 1, got {self.max_results!r}")


@dataclass(frozen=True)
class SearchResult:
    """One normalized hit."""

    title: str
    url: str
    content: str = ""


@dataclass(frozen=True)
class SearchResponse:
    """A provider-independent search result set."""

    query: str
    provider: str
    results: list[SearchResult]
    answer: str | None = None


# --------------------------------------------------------------- JSON boundary --
#: What a JSON request body may hold. ``json.dumps`` accepts exactly this, so a value it would
#: refuse cannot reach the wire.
JsonValue: TypeAlias = "str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]"

#: One decoded JSON object, straight off the wire. Its members are ``object`` until converted; the
#: accessors below are the single place that says what each one really is.
JsonObject: TypeAlias = "dict[str, object]"

#: What a query string may carry. ``cse_id`` is ``None`` for every provider but google_cse, and
#: urlencode spells that as the literal "None" -- which is what the unconfigured request already
#: sent, so it stays a value the type admits rather than a case hidden behind a cast.
QueryValue: TypeAlias = "str | int | None"


def json_object(raw: object) -> JsonObject:
    """One JSON object, with the weakest TRUE statement about its contents.

    ``isinstance(raw, dict)`` proves it is a mapping and nothing about what is in it. A provider
    that omits a block, or fills it with a scalar, reads as empty rather than raising -- the
    parsers then return no hits, which is what an empty block means."""
    return cast("JsonObject", raw) if isinstance(raw, dict) else {}


def json_array(raw: object) -> list[object]:
    """One JSON array, with the weakest TRUE statement about its contents (see :func:`json_object`)."""
    return cast("list[object]", raw) if isinstance(raw, list) else []


def json_text(block: JsonObject, key: str) -> str:
    """``block[key]`` as text. Absent, null, or empty all read as ``""``, so a provider that sends
    a field it has nothing for yields an empty string and not the word "None"."""
    value = block.get(key)
    return str(value) if value else ""


def json_answer(block: JsonObject, key: str) -> str | None:
    """``block[key]`` as an answer string, or ``None`` when the provider did not answer."""
    value = block.get(key)
    return None if value is None else str(value)


# ------------------------------------------------------------- env / selection --
def _env_key(provider: Provider) -> str | None:
    for name in _ENV_KEYS[provider]:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _configured(provider: Provider) -> bool:
    """True when ``provider`` has the credentials it needs in the environment."""
    if _env_key(provider) is None:
        return False
    if provider is Provider.GOOGLE_CSE and not os.environ.get("GOOGLE_CSE_ID"):
        return False  # the API key alone is not enough -- google_cse also needs its cx
    return True


def available_providers() -> list[Provider]:
    """The providers with a usable key in the current environment (priority order)."""
    return [p for p in Provider if _configured(p)]


def _env_hint() -> str:
    return "; ".join(f"{p.value}={'/'.join(_ENV_KEYS[p])}" for p in Provider)


def resolve_provider(config: WebSearchConfig) -> Provider:
    """Pick the provider: explicit config, then ``$HPCAGENT_BENCH_WEBSEARCH_PROVIDER``,
    then the first env-configured one. Raises when nothing is configured."""
    if config.provider is not None:
        return config.provider
    forced = os.environ.get("HPCAGENT_BENCH_WEBSEARCH_PROVIDER")
    if forced:
        try:
            return Provider(forced.strip().lower())
        except ValueError:
            raise WebSearchError(
                f"unknown web-search provider {forced!r} in "
                f"$HPCAGENT_BENCH_WEBSEARCH_PROVIDER; known: {[p.value for p in Provider]}"
            )
    for provider in available_providers():
        return provider
    raise WebSearchError("no web-search provider configured; set one of the API keys: " + _env_hint())


def _credentials(provider: Provider, config: WebSearchConfig) -> tuple[str, str | None]:
    key = config.api_key or _env_key(provider)
    if not key:
        raise WebSearchError(
            f"{provider.value}: no API key -- set {' or '.join(_ENV_KEYS[provider])} "
            f"or pass WebSearchConfig(api_key=...)"
        )
    cse_id = config.cse_id or os.environ.get("GOOGLE_CSE_ID")
    if provider is Provider.GOOGLE_CSE and not cse_id:
        raise WebSearchError("google_cse: also set GOOGLE_CSE_ID (the search-engine cx)")
    return key, cse_id


# ---------------------------------------------------------------- HTTP helpers --
def _get_request(url: str, params: dict[str, QueryValue], headers: dict[str, str]) -> urllib.request.Request:
    return urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers=headers, method="GET")


def post_request(url: str, body: dict[str, JsonValue], headers: dict[str, str]) -> urllib.request.Request:
    """Build a JSON POST ``Request`` to ``url`` (``body`` as the JSON payload,
    ``Content-Type: application/json`` merged with ``headers``). Shared by the
    per-provider request builders here and the chat agents' HTTP transport."""
    data = json.dumps(body).encode("utf-8")
    return urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST"
    )


def _http_json(request: urllib.request.Request, timeout: float) -> JsonObject:
    """The default transport: perform ``request`` and parse the JSON body, turning
    an HTTP/URL error into a :class:`WebSearchError` (never a bare stack trace).

    Every provider here answers with a JSON object; a body that decodes to anything else is
    named as such at the boundary, since the parsers downstream read it as a mapping."""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: object = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise WebSearchError(f"web search HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise WebSearchError(f"web search request failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise WebSearchError(f"web search response is a JSON {type(payload).__name__}, not an object")
    return cast("JsonObject", payload)


# ------------------------------------------------------- per-provider requests --
def _req_tavily(q: str, key: str, cse_id: str | None, cfg: WebSearchConfig) -> urllib.request.Request:
    return post_request(
        "https://api.tavily.com/search",
        {"query": q, "max_results": cfg.max_results, "include_answer": True},
        {"Authorization": f"Bearer {key}"},
    )


def _req_serper(q: str, key: str, cse_id: str | None, cfg: WebSearchConfig) -> urllib.request.Request:
    return post_request("https://google.serper.dev/search", {"q": q, "num": cfg.max_results}, {"X-API-KEY": key})


def _req_brave(q: str, key: str, cse_id: str | None, cfg: WebSearchConfig) -> urllib.request.Request:
    return _get_request(
        "https://api.search.brave.com/res/v1/web/search",
        {"q": q, "count": cfg.max_results},
        {"X-Subscription-Token": key, "Accept": "application/json"},
    )


def _req_exa(q: str, key: str, cse_id: str | None, cfg: WebSearchConfig) -> urllib.request.Request:
    return post_request("https://api.exa.ai/search", {"query": q, "numResults": cfg.max_results}, {"x-api-key": key})


def _req_google_cse(q: str, key: str, cse_id: str | None, cfg: WebSearchConfig) -> urllib.request.Request:
    return _get_request(
        "https://www.googleapis.com/customsearch/v1",
        {"key": key, "cx": cse_id, "q": q, "num": min(cfg.max_results, 10)},
        {},
    )


def _req_bing(q: str, key: str, cse_id: str | None, cfg: WebSearchConfig) -> urllib.request.Request:
    return _get_request(
        "https://api.bing.microsoft.com/v7.0/search",
        {"q": q, "count": cfg.max_results},
        {"Ocp-Apim-Subscription-Key": key},
    )


def _req_serpapi(q: str, key: str, cse_id: str | None, cfg: WebSearchConfig) -> urllib.request.Request:
    return _get_request(
        "https://serpapi.com/search.json", {"engine": "google", "q": q, "num": cfg.max_results, "api_key": key}, {}
    )


def _req_you(q: str, key: str, cse_id: str | None, cfg: WebSearchConfig) -> urllib.request.Request:
    return _get_request("https://api.ydc-index.io/search", {"query": q}, {"X-API-Key": key})


def _req_jina(q: str, key: str, cse_id: str | None, cfg: WebSearchConfig) -> urllib.request.Request:
    return _get_request(
        "https://s.jina.ai/", {"q": q}, {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    )


def _req_perplexity(q: str, key: str, cse_id: str | None, cfg: WebSearchConfig) -> urllib.request.Request:
    return post_request(
        "https://api.perplexity.ai/chat/completions",
        {"model": "sonar", "messages": [{"role": "user", "content": q}]},
        {"Authorization": f"Bearer {key}"},
    )


_REQUEST: dict[Provider, Callable[[str, str, str | None, WebSearchConfig], urllib.request.Request]] = {
    Provider.TAVILY: _req_tavily,
    Provider.SERPER: _req_serper,
    Provider.BRAVE: _req_brave,
    Provider.EXA: _req_exa,
    Provider.GOOGLE_CSE: _req_google_cse,
    Provider.BING: _req_bing,
    Provider.SERPAPI: _req_serpapi,
    Provider.YOU: _req_you,
    Provider.JINA: _req_jina,
    Provider.PERPLEXITY: _req_perplexity,
}


# --------------------------------------------------------- per-provider parsers --
#: What every parser hands back: the normalized hits, plus the provider's answer when it has one.
Parsed: TypeAlias = "tuple[list[SearchResult], str | None]"


def _hit(item: JsonObject, title_key: str, url_key: str, content_key: str) -> SearchResult:
    return SearchResult(
        title=json_text(item, title_key), url=json_text(item, url_key), content=json_text(item, content_key)
    )


def _hits(
    items: list[object], title_key: str, url_key: str, content_key: str, cfg: WebSearchConfig
) -> list[SearchResult]:
    return [_hit(json_object(it), title_key, url_key, content_key) for it in items[: cfg.max_results]]


def _parse_tavily(data: JsonObject, cfg: WebSearchConfig) -> Parsed:
    return _hits(json_array(data.get("results")), "title", "url", "content", cfg), json_answer(data, "answer")


def _parse_serper(data: JsonObject, cfg: WebSearchConfig) -> Parsed:
    hits = _hits(json_array(data.get("organic")), "title", "link", "snippet", cfg)
    return hits, json_answer(json_object(data.get("answerBox")), "answer")


def _parse_brave(data: JsonObject, cfg: WebSearchConfig) -> Parsed:
    results = json_array(json_object(data.get("web")).get("results"))
    return _hits(results, "title", "url", "description", cfg), None


def _parse_exa(data: JsonObject, cfg: WebSearchConfig) -> Parsed:
    results: list[SearchResult] = []
    for raw in json_array(data.get("results"))[: cfg.max_results]:
        item = json_object(raw)
        content = json_text(item, "text") or json_text(item, "snippet")
        results.append(SearchResult(title=json_text(item, "title"), url=json_text(item, "url"), content=content))
    return results, None


def _parse_google_cse(data: JsonObject, cfg: WebSearchConfig) -> Parsed:
    return _hits(json_array(data.get("items")), "title", "link", "snippet", cfg), None


def _parse_bing(data: JsonObject, cfg: WebSearchConfig) -> Parsed:
    results = json_array(json_object(data.get("webPages")).get("value"))
    return _hits(results, "name", "url", "snippet", cfg), None


def _parse_serpapi(data: JsonObject, cfg: WebSearchConfig) -> Parsed:
    hits = _hits(json_array(data.get("organic_results")), "title", "link", "snippet", cfg)
    return hits, json_answer(json_object(data.get("answer_box")), "answer")


def _parse_you(data: JsonObject, cfg: WebSearchConfig) -> Parsed:
    results: list[SearchResult] = []
    for raw in json_array(data.get("hits"))[: cfg.max_results]:
        item = json_object(raw)
        snippets = item.get("snippets")
        passages = json_array(snippets)
        # you.com sends the body as a list of passages, and an EMPTY list is still that answer (no
        # body). A hit that sends no list at all carries a plain description instead.
        joined = " ".join(str(part) for part in passages)
        content = joined if isinstance(snippets, list) else json_text(item, "description")
        results.append(SearchResult(title=json_text(item, "title"), url=json_text(item, "url"), content=content))
    return results, None


def _parse_jina(data: JsonObject, cfg: WebSearchConfig) -> Parsed:
    return _hits(json_array(data.get("data")), "title", "url", "content", cfg), None


def _parse_perplexity(data: JsonObject, cfg: WebSearchConfig) -> Parsed:
    choices = json_array(data.get("choices"))
    message = json_object(json_object(choices[0]).get("message")) if choices else {}
    citations = json_array(data.get("citations"))
    results = [SearchResult(title="", url=str(u), content="") for u in citations[: cfg.max_results]]
    return results, json_text(message, "content") or None


_PARSE: dict[Provider, Callable[[JsonObject, WebSearchConfig], Parsed]] = {
    Provider.TAVILY: _parse_tavily,
    Provider.SERPER: _parse_serper,
    Provider.BRAVE: _parse_brave,
    Provider.EXA: _parse_exa,
    Provider.GOOGLE_CSE: _parse_google_cse,
    Provider.BING: _parse_bing,
    Provider.SERPAPI: _parse_serpapi,
    Provider.YOU: _parse_you,
    Provider.JINA: _parse_jina,
    Provider.PERPLEXITY: _parse_perplexity,
}


# ----------------------------------------------------------------- public entry --
def search(
    query: str,
    config: WebSearchConfig | None = None,
    *,
    transport: Callable[[urllib.request.Request], JsonObject] | None = None,
) -> SearchResponse:
    """Search ``query`` and return a normalized :class:`SearchResponse`.

    ``config`` selects the provider + limits (default: auto-detect, 5 results).
    ``transport`` (a ``Request -> dict`` callable) overrides the HTTP layer -- the
    seam that lets tests drive every provider with no network.
    """
    config = config or WebSearchConfig()
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    provider = resolve_provider(config)
    key, cse_id = _credentials(provider, config)
    request = _REQUEST[provider](query, key, cse_id, config)
    transport = transport or (lambda req: _http_json(req, config.timeout))
    data = transport(request)
    results, answer = _PARSE[provider](data, config)
    return SearchResponse(query=query, provider=provider.value, results=results, answer=answer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hpcagent_bench.websearch", description="Provider-agnostic web search.")
    parser.add_argument("query", nargs="?", help="the search query")
    parser.add_argument("--provider", choices=[p.value for p in Provider], help="force a provider (else auto-detect)")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--list", action="store_true", help="list providers configured in this environment and exit")
    parser.add_argument("--json", action="store_true", help="emit the raw normalized JSON")
    args = parser.parse_args(argv)

    if args.list:
        found = available_providers()
        print("configured providers: " + (", ".join(p.value for p in found) if found else "(none)"))
        if not found:
            print("set an API key, e.g. TAVILY_API_KEY=...", file=sys.stderr)
        return 0
    if not args.query:
        parser.error("a query is required (or use --list)")

    config = WebSearchConfig(provider=Provider(args.provider) if args.provider else None, max_results=args.max_results)
    try:
        response = search(args.query, config)
    except WebSearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(dataclasses.asdict(response), indent=2))
    else:
        print(f"[{response.provider}] {len(response.results)} result(s) for {response.query!r}")
        if response.answer:
            print(f"\nanswer: {response.answer}\n")
        for i, r in enumerate(response.results, 1):
            print(f"{i}. {r.title}\n   {r.url}\n   {r.content[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
