"""Async SerpAPI and OpenAI-backed web research helpers."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from optarena.tools.optarena_tools.config import settings
from optarena.tools.optarena_tools.schemas import (
    Citation,
    ResearchRequest,
    ResearchResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

SERPAPI_URL = "https://serpapi.com/search.json"


class ToolConfigurationError(RuntimeError):
    """Raised when a requested tool is missing its required environment."""


async def serpapi_search(request: SearchRequest) -> SearchResponse:
    if not settings.serpapi_api_key:
        raise ToolConfigurationError("SERPAPI_API_KEY is required for /web/search")

    params: dict[str, Any] = {
        "engine": "google",
        "q": request.query,
        "api_key": settings.serpapi_api_key,
        "num": request.num_results,
        "hl": request.language,
    }
    if request.location:
        params["location"] = request.location

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(SERPAPI_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    results = []
    for item in payload.get("organic_results", [])[:request.num_results]:
        results.append(
            SearchResult(
                title=item.get("title") or "",
                link=item.get("link") or "",
                snippet=item.get("snippet") or item.get("rich_snippet", {}).get("top", {}).get("extensions", [""])[0],
                position=item.get("position"),
            )
        )
    return SearchResponse(query=request.query, results=results, raw=payload)


def _default_search_queries(query: str) -> list[str]:
    return [query]


async def _plan_search_queries_with_openai(query: str) -> list[str]:
    if not settings.openai_api_key:
        return _default_search_queries(query)

    payload = {
        "model": settings.openai_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Create concise Google search queries for web research. "
                    "Return only JSON with this shape: {\"queries\": [\"...\"]}."
                ),
            },
            {"role": "user", "content": f"Research question: {query}\nCreate 2-4 complementary searches."},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{settings.openai_base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        queries = [str(item).strip() for item in parsed.get("queries", []) if str(item).strip()]
        return queries[:4] or _default_search_queries(query)
    except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError):
        return _default_search_queries(query)


def _citation_lines(citations: list[Citation]) -> str:
    lines = []
    for idx, citation in enumerate(citations, start=1):
        lines.append(f"[{idx}] {citation.title}\nURL: {citation.link}\nSnippet: {citation.snippet}")
    return "\n\n".join(lines)


async def _summarize_with_openai(query: str, citations: list[Citation]) -> str:
    if not settings.openai_api_key:
        return _fallback_answer(query, citations)

    prompt = (
        "Answer the user using only the cited search results. "
        "Keep it concise and cite sources inline as [1], [2].\n\n"
        f"User question: {query}\n\nSearch results:\n{_citation_lines(citations)}"
    )
    payload = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": "You are a concise web research assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{settings.openai_base_url}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"]


def _fallback_answer(query: str, citations: list[Citation]) -> str:
    if not citations:
        return f"No search results were found for: {query}"
    top = citations[:5]
    bullets = [f"- [{idx}] {item.title}: {item.snippet}" for idx, item in enumerate(top, start=1)]
    return "OpenAI summarization is disabled or not configured. Top cited results:\n" + "\n".join(bullets)


async def _run_external_research_agent(request: ResearchRequest) -> ResearchResponse:
    if not settings.web_research_agent_path:
        raise ToolConfigurationError("WEB_RESEARCH_AGENT_PATH is not configured")
    script = Path(settings.web_research_agent_path)
    if not script.exists():
        raise ToolConfigurationError(f"WEB_RESEARCH_AGENT_PATH does not exist: {script}")

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        "-q",
        request.query,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=settings.web_research_agent_timeout_seconds)
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.communicate()
        raise TimeoutError("external web research agent timed out") from exc

    out = stdout.decode("utf-8", errors="replace").strip()
    err = stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"external web research agent failed with {proc.returncode}: {err}")

    trace: dict[str, Any] = {"mode": "external", "returncode": proc.returncode, "stderr": err}
    try:
        trace["json"] = json.loads(out)
    except json.JSONDecodeError:
        trace["stdout"] = out

    return ResearchResponse(query=request.query, answer=out, citations=[], trace=trace)


async def research(request: ResearchRequest) -> ResearchResponse:
    if request.use_external_agent:
        return await _run_external_research_agent(request)

    queries = request.search_queries or await _plan_search_queries_with_openai(request.query)
    tasks = [
        serpapi_search(SearchRequest(query=query, num_results=request.num_results_per_query))
        for query in queries
    ]
    search_responses = await asyncio.gather(*tasks)

    citations: list[Citation] = []
    seen_links: set[str] = set()
    for search_response in search_responses:
        for result in search_response.results:
            link = str(result.link)
            if link in seen_links:
                continue
            seen_links.add(link)
            citations.append(Citation(title=result.title, link=result.link, snippet=result.snippet))

    answer = await _summarize_with_openai(request.query, citations) if request.summarize else _fallback_answer(
        request.query, citations)
    return ResearchResponse(
        query=request.query,
        answer=answer,
        citations=citations,
        trace={
            "mode": "builtin",
            "search_queries": queries,
            "planned_with_llm": request.search_queries is None and bool(settings.openai_api_key),
            "num_searches": len(queries),
            "num_citations": len(citations),
            "summarized": request.summarize and bool(settings.openai_api_key),
        },
    )
