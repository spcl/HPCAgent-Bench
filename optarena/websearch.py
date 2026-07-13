# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Provider-agnostic web search -- one call, any popular backend, keyed by env var.

A thin, stdlib-only client (``urllib``, no third-party dep) so an agent (or the
harness) can search the web through whichever provider the environment has a key
for. The provider is picked, in order, from:

1. an explicit :class:`WebSearchConfig` ``provider``;
2. ``$OPTARENA_WEBSEARCH_PROVIDER``;
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

    python -m optarena.websearch "fast gemm avx512" --max-results 5
    python -m optarena.websearch --list          # which providers have a key here
"""
import argparse
import dataclasses
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional


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
_ENV_KEYS: Dict[Provider, tuple] = {
    Provider.TAVILY: ("TAVILY_API_KEY", ),
    Provider.SERPER: ("SERPER_API_KEY", ),
    Provider.BRAVE: ("BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY"),
    Provider.EXA: ("EXA_API_KEY", ),
    Provider.GOOGLE_CSE: ("GOOGLE_CSE_API_KEY", ),
    Provider.BING: ("BING_SEARCH_API_KEY", "BING_SUBSCRIPTION_KEY"),
    Provider.SERPAPI: ("SERPAPI_API_KEY", "SERPAPI_KEY"),
    Provider.YOU: ("YDC_API_KEY", "YOU_API_KEY"),
    Provider.JINA: ("JINA_API_KEY", ),
    Provider.PERPLEXITY: ("PERPLEXITY_API_KEY", ),
}


class WebSearchError(RuntimeError):
    """A configuration or transport failure in a web-search call."""


@dataclass(frozen=True)
class WebSearchConfig:
    """How to search -- a config object, never a bag of positional strings.

    ``provider`` ``None`` auto-detects from the environment. ``api_key`` / ``cse_id``
    override the env-resolved credentials (e.g. to pass a key held elsewhere).
    """
    provider: Optional[Provider] = None
    max_results: int = 5
    timeout: float = 30.0
    api_key: Optional[str] = None
    cse_id: Optional[str] = None  # google_cse only (the search-engine cx)

    def __post_init__(self):
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
    results: List[SearchResult]
    answer: Optional[str] = None


# ------------------------------------------------------------- env / selection --
def _env_key(provider: Provider) -> Optional[str]:
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


def available_providers() -> List[Provider]:
    """The providers with a usable key in the current environment (priority order)."""
    return [p for p in Provider if _configured(p)]


def _env_hint() -> str:
    return "; ".join(f"{p.value}={'/'.join(_ENV_KEYS[p])}" for p in Provider)


def resolve_provider(config: WebSearchConfig) -> Provider:
    """Pick the provider: explicit config, then ``$OPTARENA_WEBSEARCH_PROVIDER``,
    then the first env-configured one. Raises when nothing is configured."""
    if config.provider is not None:
        return config.provider
    forced = os.environ.get("OPTARENA_WEBSEARCH_PROVIDER")
    if forced:
        return Provider(forced.strip().lower())
    for provider in available_providers():
        return provider
    raise WebSearchError("no web-search provider configured; set one of the API keys: " + _env_hint())


def _credentials(provider: Provider, config: WebSearchConfig) -> tuple:
    key = config.api_key or _env_key(provider)
    if not key:
        raise WebSearchError(f"{provider.value}: no API key -- set {' or '.join(_ENV_KEYS[provider])} "
                             f"or pass WebSearchConfig(api_key=...)")
    cse_id = config.cse_id or os.environ.get("GOOGLE_CSE_ID")
    if provider is Provider.GOOGLE_CSE and not cse_id:
        raise WebSearchError("google_cse: also set GOOGLE_CSE_ID (the search-engine cx)")
    return key, cse_id


# ---------------------------------------------------------------- HTTP helpers --
def _get_request(url: str, params: dict, headers: dict) -> urllib.request.Request:
    return urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers=headers, method="GET")


def _post_request(url: str, body: dict, headers: dict) -> urllib.request.Request:
    data = json.dumps(body).encode("utf-8")
    return urllib.request.Request(url,
                                  data=data,
                                  headers={
                                      "Content-Type": "application/json",
                                      **headers
                                  },
                                  method="POST")


def _http_json(request: urllib.request.Request, timeout: float) -> dict:
    """The default transport: perform ``request`` and parse the JSON body, turning
    an HTTP/URL error into a :class:`WebSearchError` (never a bare stack trace)."""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise WebSearchError(f"web search HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise WebSearchError(f"web search request failed: {exc}") from exc


# ------------------------------------------------------- per-provider requests --
def _req_tavily(q, key, cse_id, cfg):
    return _post_request("https://api.tavily.com/search", {
        "query": q,
        "max_results": cfg.max_results,
        "include_answer": True
    }, {"Authorization": f"Bearer {key}"})


def _req_serper(q, key, cse_id, cfg):
    return _post_request("https://google.serper.dev/search", {"q": q, "num": cfg.max_results}, {"X-API-KEY": key})


def _req_brave(q, key, cse_id, cfg):
    return _get_request("https://api.search.brave.com/res/v1/web/search", {
        "q": q,
        "count": cfg.max_results
    }, {
        "X-Subscription-Token": key,
        "Accept": "application/json"
    })


def _req_exa(q, key, cse_id, cfg):
    return _post_request("https://api.exa.ai/search", {"query": q, "numResults": cfg.max_results}, {"x-api-key": key})


def _req_google_cse(q, key, cse_id, cfg):
    return _get_request("https://www.googleapis.com/customsearch/v1", {
        "key": key,
        "cx": cse_id,
        "q": q,
        "num": min(cfg.max_results, 10)
    }, {})


def _req_bing(q, key, cse_id, cfg):
    return _get_request("https://api.bing.microsoft.com/v7.0/search", {
        "q": q,
        "count": cfg.max_results
    }, {"Ocp-Apim-Subscription-Key": key})


def _req_serpapi(q, key, cse_id, cfg):
    return _get_request("https://serpapi.com/search.json", {
        "engine": "google",
        "q": q,
        "num": cfg.max_results,
        "api_key": key
    }, {})


def _req_you(q, key, cse_id, cfg):
    return _get_request("https://api.ydc-index.io/search", {"query": q}, {"X-API-Key": key})


def _req_jina(q, key, cse_id, cfg):
    return _get_request("https://s.jina.ai/", {"q": q}, {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json"
    })


def _req_perplexity(q, key, cse_id, cfg):
    return _post_request("https://api.perplexity.ai/chat/completions", {
        "model": "sonar",
        "messages": [{
            "role": "user",
            "content": q
        }]
    }, {"Authorization": f"Bearer {key}"})


_REQUEST: Dict[Provider, Callable] = {
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
def _hit(item: dict, title_key: str, url_key: str, content_key: str) -> SearchResult:
    return SearchResult(title=str(item.get(title_key, "") or ""),
                        url=str(item.get(url_key, "") or ""),
                        content=str(item.get(content_key, "") or ""))


def _hits(items, title_key, url_key, content_key, cfg) -> List[SearchResult]:
    return [_hit(it, title_key, url_key, content_key) for it in (items or [])[:cfg.max_results]]


def _parse_tavily(data, cfg):
    return _hits(data.get("results"), "title", "url", "content", cfg), data.get("answer")


def _parse_serper(data, cfg):
    return _hits(data.get("organic"), "title", "link", "snippet", cfg), (data.get("answerBox") or {}).get("answer")


def _parse_brave(data, cfg):
    return _hits((data.get("web") or {}).get("results"), "title", "url", "description", cfg), None


def _parse_exa(data, cfg):
    items = data.get("results") or []
    results = [
        SearchResult(title=str(it.get("title", "") or ""),
                     url=str(it.get("url", "") or ""),
                     content=str(it.get("text", "") or it.get("snippet", "") or "")) for it in items[:cfg.max_results]
    ]
    return results, None


def _parse_google_cse(data, cfg):
    return _hits(data.get("items"), "title", "link", "snippet", cfg), None


def _parse_bing(data, cfg):
    return _hits((data.get("webPages") or {}).get("value"), "name", "url", "snippet", cfg), None


def _parse_serpapi(data, cfg):
    return _hits(data.get("organic_results"), "title", "link", "snippet", cfg), (data.get("answer_box")
                                                                                 or {}).get("answer")


def _parse_you(data, cfg):
    items = data.get("hits") or []
    results = []
    for it in items[:cfg.max_results]:
        snippets = it.get("snippets")
        content = " ".join(snippets) if isinstance(snippets, list) else str(it.get("description", "") or "")
        results.append(
            SearchResult(title=str(it.get("title", "") or ""), url=str(it.get("url", "") or ""), content=content))
    return results, None


def _parse_jina(data, cfg):
    return _hits(data.get("data"), "title", "url", "content", cfg), None


def _parse_perplexity(data, cfg):
    choices = data.get("choices") or []
    answer = (choices[0].get("message") or {}).get("content", "") if choices else ""
    citations = data.get("citations") or []
    results = [SearchResult(title="", url=str(u), content="") for u in citations[:cfg.max_results]]
    return results, answer or None


_PARSE: Dict[Provider, Callable] = {
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
def search(query: str,
           config: Optional[WebSearchConfig] = None,
           *,
           transport: Optional[Callable[[urllib.request.Request], dict]] = None) -> SearchResponse:
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="optarena.websearch", description="Provider-agnostic web search.")
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
