# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Provider-agnostic web search (:mod:`optarena.websearch`): env-keyed provider
selection, per-provider request shaping, and one normalized result shape -- all
driven with an injected transport, so no test ever touches the network."""
import json
from urllib.parse import parse_qs, urlparse

import pytest

from optarena import websearch
from optarena.websearch import Provider, SearchResponse, WebSearchConfig, WebSearchError, search

#: All env vars any provider reads -- cleared before each test so the host's real
#: keys never leak into selection assertions.
_ALL_KEYS = sorted({k
                    for keys in websearch._ENV_KEYS.values()
                    for k in keys} | {"OPTARENA_WEBSEARCH_PROVIDER", "GOOGLE_CSE_ID"})


@pytest.fixture(autouse=True)
def _clean_search_env(monkeypatch):
    for k in _ALL_KEYS:
        monkeypatch.delenv(k, raising=False)


def _headers(req):
    """Case-insensitive header map (urllib capitalizes stored header keys)."""
    return {k.lower(): v for k, v in req.header_items()}


def _query(req):
    return parse_qs(urlparse(req.full_url).query)


def _body(req):
    return json.loads(req.data.decode("utf-8"))


# --- the config dataclass -----------------------------------------------------


def test_config_coerces_and_validates():
    assert WebSearchConfig(provider="tavily").provider is Provider.TAVILY  # string coerced to enum
    assert WebSearchConfig().provider is None  # auto-detect
    with pytest.raises(ValueError):
        WebSearchConfig(provider="altavista")  # unknown provider
    with pytest.raises(ValueError):
        WebSearchConfig(max_results=0)


# --- provider selection -------------------------------------------------------


def test_explicit_config_provider_wins(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "T")  # even with tavily configured ...
    assert websearch.resolve_provider(WebSearchConfig(provider="brave")) is Provider.BRAVE  # ... explicit wins


def test_env_override_selects_provider(monkeypatch):
    monkeypatch.setenv("OPTARENA_WEBSEARCH_PROVIDER", "serper")
    assert websearch.resolve_provider(WebSearchConfig()) is Provider.SERPER


def test_autodetect_follows_declaration_priority(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "B")
    monkeypatch.setenv("SERPER_API_KEY", "S")  # serper is declared before brave -> wins
    assert websearch.resolve_provider(WebSearchConfig()) is Provider.SERPER
    assert websearch.available_providers() == [Provider.SERPER, Provider.BRAVE]


def test_brave_second_env_var_is_accepted(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "B")  # the alternate key name
    assert Provider.BRAVE in websearch.available_providers()


def test_google_cse_needs_both_key_and_cx(monkeypatch):
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "K")
    assert Provider.GOOGLE_CSE not in websearch.available_providers()  # key alone is not enough
    monkeypatch.setenv("GOOGLE_CSE_ID", "CX")
    assert Provider.GOOGLE_CSE in websearch.available_providers()


def test_no_provider_configured_raises():
    with pytest.raises(WebSearchError, match="no web-search provider"):
        websearch.resolve_provider(WebSearchConfig())


def test_missing_api_key_raises():
    with pytest.raises(WebSearchError, match="no API key"):
        search("q", WebSearchConfig(provider="tavily"))  # provider forced, but no key anywhere


def test_empty_query_raises():
    with pytest.raises(ValueError, match="non-empty"):
        search("   ", WebSearchConfig(provider="tavily", api_key="K"), transport=lambda req: {})


# --- request shaping (representative providers) -------------------------------


def _capture(provider, canned=None):
    box = {}

    def transport(req):
        box["req"] = req
        return canned or {}

    search("gemm avx512", WebSearchConfig(provider=provider, api_key="K", cse_id="CX"), transport=transport)
    return box["req"]


def test_tavily_request_is_post_with_bearer_and_query():
    req = _capture(Provider.TAVILY)
    assert req.full_url == "https://api.tavily.com/search" and req.method == "POST"
    assert _body(req)["query"] == "gemm avx512" and _body(req)["max_results"] == 5
    assert _headers(req)["authorization"] == "Bearer K"


def test_serper_request_sends_api_key_header():
    req = _capture(Provider.SERPER)
    assert req.method == "POST" and _headers(req)["x-api-key"] == "K"
    assert _body(req)["q"] == "gemm avx512"


def test_brave_request_is_get_with_subscription_token():
    req = _capture(Provider.BRAVE)
    assert req.method == "GET" and _headers(req)["x-subscription-token"] == "K"
    assert _query(req)["q"] == ["gemm avx512"] and _query(req)["count"] == ["5"]


def test_google_cse_request_carries_key_and_cx():
    req = _capture(Provider.GOOGLE_CSE)
    q = _query(req)
    assert q["key"] == ["K"] and q["cx"] == ["CX"] and q["q"] == ["gemm avx512"]


def test_perplexity_request_is_a_chat_completion():
    req = _capture(Provider.PERPLEXITY)
    assert req.full_url == "https://api.perplexity.ai/chat/completions"
    assert _body(req)["messages"][0]["content"] == "gemm avx512"


# --- response normalization (every provider -> one shape) ---------------------

_CANNED = {
    Provider.TAVILY: ({
        "results": [{
            "title": "T",
            "url": "http://a",
            "content": "C"
        }],
        "answer": "ANS"
    }, "ANS"),
    Provider.SERPER: ({
        "organic": [{
            "title": "T",
            "link": "http://a",
            "snippet": "C"
        }],
        "answerBox": {
            "answer": "ANS"
        }
    }, "ANS"),
    Provider.BRAVE: ({
        "web": {
            "results": [{
                "title": "T",
                "url": "http://a",
                "description": "C"
            }]
        }
    }, None),
    Provider.EXA: ({
        "results": [{
            "title": "T",
            "url": "http://a",
            "text": "C"
        }]
    }, None),
    Provider.GOOGLE_CSE: ({
        "items": [{
            "title": "T",
            "link": "http://a",
            "snippet": "C"
        }]
    }, None),
    Provider.BING: ({
        "webPages": {
            "value": [{
                "name": "T",
                "url": "http://a",
                "snippet": "C"
            }]
        }
    }, None),
    Provider.SERPAPI: ({
        "organic_results": [{
            "title": "T",
            "link": "http://a",
            "snippet": "C"
        }],
        "answer_box": {
            "answer": "ANS"
        }
    }, "ANS"),
    Provider.YOU: ({
        "hits": [{
            "title": "T",
            "url": "http://a",
            "snippets": ["C", "extra"]
        }]
    }, None),
    Provider.JINA: ({
        "data": [{
            "title": "T",
            "url": "http://a",
            "content": "C"
        }]
    }, None),
    Provider.PERPLEXITY: ({
        "choices": [{
            "message": {
                "content": "ANS"
            }
        }],
        "citations": ["http://a"]
    }, "ANS"),
}


@pytest.mark.parametrize("provider", list(Provider))
def test_every_provider_normalizes_to_one_shape(provider):
    canned, expect_answer = _CANNED[provider]
    resp = search("q", WebSearchConfig(provider=provider, api_key="K", cse_id="CX"), transport=lambda req: canned)
    assert isinstance(resp, SearchResponse) and resp.provider == provider.value
    assert len(resp.results) == 1
    hit = resp.results[0]
    assert hit.url == "http://a"
    if provider is Provider.PERPLEXITY:
        assert hit.title == "" and hit.content == ""  # an answer engine -> citations only
    else:
        assert hit.title == "T" and hit.content.startswith("C")
    assert resp.answer == expect_answer


def test_max_results_trims_the_list():
    many = {"results": [{"title": f"t{i}", "url": f"http://{i}", "content": ""} for i in range(20)]}
    resp = search("q", WebSearchConfig(provider="tavily", api_key="K", max_results=3), transport=lambda req: many)
    assert len(resp.results) == 3


def test_api_key_override_bypasses_env():
    """A key on the config is used even with nothing in the environment."""
    resp = search("q",
                  WebSearchConfig(provider="tavily", api_key="explicit"),
                  transport=lambda req: {
                      "results": [],
                      "answer": None
                  })
    assert resp.provider == "tavily" and resp.results == []


# --- CLI ----------------------------------------------------------------------


def test_cli_list_reports_configured(capsys, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "T")
    assert websearch.main(["--list"]) == 0
    assert "tavily" in capsys.readouterr().out


def test_cli_errors_when_nothing_configured(capsys):
    assert websearch.main(["some query"]) == 1  # no provider key in the (cleaned) env
    assert "error" in capsys.readouterr().err.lower()
