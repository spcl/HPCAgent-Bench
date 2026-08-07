#!/usr/bin/env python3
"""Process-oriented WebSearch tool for the future judge service.

The tool is intentionally service-agnostic:
  1. read a query and environment configuration,
  2. call SerpAPI,
  3. crawl selected pages with Crawl4AI,
  4. ask an OpenAI/vLLM-compatible LLM endpoint to synthesize an answer,
  5. print JSON to stdout.

A later HTTP service can either spawn this file per request or import run_web_search().
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import json
import os
import pathlib
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class CrawledPage:
    title: str
    url: str
    snippet: str
    content: str
    references: str
    success: bool
    error: str = ""


DEBUG = False


def debug(message: str) -> None:
    if DEBUG:
        print(f"[web_search] {message}", file=sys.stderr, flush=True)


def load_env_file(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def discover_env_file(explicit: Optional[str]) -> None:
    candidates: List[pathlib.Path] = []
    if explicit:
        candidates.append(pathlib.Path(explicit))
    candidates.append(pathlib.Path.cwd() / ".env")
    candidates.append(pathlib.Path(__file__).resolve().parents[1] / ".env")
    for path in candidates:
        if path.exists():
            load_env_file(path)
            return


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def post_json(url: str,
              payload: Dict[str, Any],
              timeout: float,
              headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def retryable_post_json(url: str, payload: Dict[str, Any], timeout: float, headers: Dict[str, str]) -> Dict[str, Any]:
    try:
        return post_json(url, payload, timeout, headers)
    except RuntimeError as exc:
        text = str(exc)
        if "temperature" in text and "Unsupported value" in text and "temperature" in payload:
            payload.pop("temperature", None)
            return post_json(url, payload, timeout, headers)
        if "Unsupported parameter" in text and "reasoning_effort" in text and "reasoning_effort" in payload:
            payload.pop("reasoning_effort", None)
            return post_json(url, payload, timeout, headers)
        if "Unsupported parameter" in text and "verbosity" in text and "verbosity" in payload:
            payload.pop("verbosity", None)
            return post_json(url, payload, timeout, headers)
        if "Unsupported parameter" in text and "max_tokens" in text and "max_completion_tokens" in text:
            value = payload.pop("max_tokens", None)
            if value is not None:
                payload["max_completion_tokens"] = value
                return post_json(url, payload, timeout, headers)
        if "Unsupported parameter" in text and "max_completion_tokens" in text and "max_tokens" in text:
            value = payload.pop("max_completion_tokens", None)
            if value is not None:
                payload["max_tokens"] = value
                return post_json(url, payload, timeout, headers)
        raise


def get_json(url: str, params: Dict[str, str], timeout: float) -> Dict[str, Any]:
    query = urllib.parse.urlencode(params)
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(f"{url}{sep}{query}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_serpapi(query: str, max_results: int, timeout: float) -> List[SearchResult]:
    api_key = os.environ.get("SERPAPI_API_KEY", "").strip()
    serpapi_url = os.environ.get("SERPAPI_URL", "https://serpapi.com/search.json").strip()
    if not api_key and "localhost" not in serpapi_url and "127.0.0.1" not in serpapi_url:
        raise RuntimeError("SERPAPI_API_KEY must be set")

    debug(f"calling SerpAPI url={serpapi_url} max_results={max_results}")
    payload = get_json(serpapi_url, {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": str(max_results),
    }, timeout)

    organic = payload.get("organic_results") or payload.get("results") or []
    results: List[SearchResult] = []
    for item in organic:
        url = str(item.get("link") or item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            SearchResult(title=str(item.get("title") or url),
                         url=url,
                         snippet=str(item.get("snippet") or item.get("description") or "")))
        if len(results) >= max_results:
            break
    debug(f"SerpAPI returned {len(results)} usable URLs")
    return results


def markdown_text(markdown: Any) -> str:
    for attr in ("fit_markdown", "markdown_with_citations", "raw_markdown"):
        value = getattr(markdown, attr, None)
        if value:
            return str(value)
    return str(markdown or "")


def markdown_references(markdown: Any) -> str:
    return str(getattr(markdown, "references_markdown", "") or "")


async def collect_arun_many(crawler: Any, urls: List[str], config: Any) -> List[Any]:
    crawled = crawler.arun_many(urls, config=config)
    if inspect.isawaitable(crawled):
        crawled = await crawled
    if hasattr(crawled, "__aiter__"):
        return [item async for item in crawled]
    return list(crawled)


async def crawl_with_crawl4ai(results: Iterable[SearchResult], query: str, max_pages: int,
                              max_chars: int) -> List[CrawledPage]:
    fake = os.environ.get("WEBSEARCH_FAKE_CRAWL_JSON", "").strip()
    if fake:
        debug("using WEBSEARCH_FAKE_CRAWL_JSON instead of Crawl4AI")
        mapping = json.loads(fake)
        pages = []
        for result in list(results)[:max_pages]:
            content = str(mapping.get(result.url, ""))
            pages.append(
                CrawledPage(result.title, result.url, result.snippet, content[:max_chars], "", bool(content),
                            "" if content else "no fake crawl content for url"))
        return pages

    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import BM25ContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "crawl4ai is required for live crawling; install requirements.txt and playwright chromium") from exc

    bm25_threshold = env_float("WEBSEARCH_BM25_THRESHOLD", 1.0)
    page_timeout = env_int("WEBSEARCH_PAGE_TIMEOUT_MS", 30000)
    semaphore_count = env_int("WEBSEARCH_CRAWL_CONCURRENCY", min(max(max_pages, 1), 3))
    check_robots = env_bool("WEBSEARCH_CHECK_ROBOTS_TXT", True)
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=BM25ContentFilter(
            user_query=query,
            bm25_threshold=bm25_threshold,
            language=os.environ.get("WEBSEARCH_BM25_LANGUAGE", "english"),
        ),
        options={
            "citations": True,
            "ignore_images": True,
            "body_width": 0,
        },
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=markdown_generator,
        page_timeout=page_timeout,
        check_robots_txt=check_robots,
        excluded_tags=["nav", "aside", "footer", "form", "script", "style"],
        semaphore_count=semaphore_count,
    )
    browser_config = BrowserConfig(headless=True, verbose=False)
    pages: List[CrawledPage] = []
    selected = list(results)[:max_pages]
    debug(f"crawling {len(selected)} page(s) with Crawl4AI")
    by_url = {result.url: result for result in selected}
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                crawl_results = await collect_arun_many(crawler, [result.url for result in selected], run_config)
        except Exception:
            crawl_results = []
            for result in selected:
                with contextlib.redirect_stdout(sys.stderr):
                    crawl_results.append(await crawler.arun(url=result.url, config=run_config))

    seen_urls = set()
    for crawled in crawl_results:
        url = str(getattr(crawled, "url", "") or "")
        source = by_url.get(url) or next((candidate for candidate in selected if candidate.url == url), None)
        if source is None:
            source = SearchResult(title=url or "unknown", url=url, snippet="")
        seen_urls.add(source.url)
        if getattr(crawled, "success", False):
            markdown = getattr(crawled, "markdown", "")
            content = markdown_text(markdown)[:max_chars]
            references = markdown_references(markdown)
            pages.append(CrawledPage(source.title, source.url, source.snippet, content, references, True))
        else:
            pages.append(
                CrawledPage(source.title, source.url, source.snippet, "", "", False,
                            str(getattr(crawled, "error_message", "crawl failed"))))

    for result in selected:
        if result.url not in seen_urls:
            pages.append(
                CrawledPage(result.title, result.url, result.snippet, "", "", False, "crawl did not return a result"))
    debug(f"Crawl4AI returned {sum(1 for page in pages if page.success)} successful page(s)")
    return pages


def build_llm_messages(query: str, pages: List[CrawledPage]) -> List[Dict[str, str]]:
    context_blocks = []
    for idx, page in enumerate(pages, 1):
        if page.content:
            context_blocks.append(
                textwrap.dedent(f"""
            [Source {idx}]
            Title: {page.title}
            URL: {page.url}
            Snippet: {page.snippet}
            Content:
            {page.content}
            References:
            {page.references}
            """).strip())
    context = "\n\n".join(context_blocks) if context_blocks else "No crawled page content was available."
    return [
        {
            "role":
            "system",
            "content": ("Answer the user's web search query using only the provided sources. "
                        "Be concise, technical, and include source URLs in a short Sources section."),
        },
        {
            "role": "user",
            "content": f"Query: {query}\n\nSources:\n{context}",
        },
    ]


def call_llm(query: str, pages: List[CrawledPage], timeout: float) -> str:
    base = os.environ.get("WEBSEARCH_LLM_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("WEBSEARCH_LLM_MODEL", "").strip()
    if not base:
        raise RuntimeError("WEBSEARCH_LLM_BASE_URL must be set")
    if not model:
        raise RuntimeError("WEBSEARCH_LLM_MODEL must be set")

    headers: Dict[str, str] = {}
    api_key = os.environ.get("WEBSEARCH_LLM_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    token_field = os.environ.get("WEBSEARCH_LLM_TOKEN_FIELD", "max_tokens").strip() or "max_tokens"
    max_tokens = env_int("WEBSEARCH_LLM_MAX_TOKENS", 4096)
    payload = {
        "model": model,
        "messages": build_llm_messages(query, pages),
        token_field: max_tokens,
    }
    reasoning_effort = os.environ.get("WEBSEARCH_LLM_REASONING_EFFORT", "minimal").strip()
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    verbosity = os.environ.get("WEBSEARCH_LLM_VERBOSITY", "").strip()
    if verbosity:
        payload["verbosity"] = verbosity
    temperature = os.environ.get("WEBSEARCH_LLM_TEMPERATURE", "").strip()
    if temperature:
        payload["temperature"] = float(temperature)
    debug(f"calling LLM url={base}/chat/completions model={model} pages={len(pages)}")
    url = f"{base}/chat/completions"
    response = retryable_post_json(url, payload, timeout, headers)
    try:
        answer = str(response["choices"][0]["message"]["content"] or "")
        if not answer.strip():
            finish_reason = response["choices"][0].get("finish_reason")
            if finish_reason == "length":
                retry_multiplier = env_int("WEBSEARCH_LLM_EMPTY_RETRY_MULTIPLIER", 4)
                current_limit = payload.get("max_completion_tokens", payload.get("max_tokens", max_tokens))
                retry_limit = int(current_limit) * retry_multiplier
                debug(f"LLM returned empty length-limited response; retrying with output limit={retry_limit}")
                if "max_completion_tokens" in payload:
                    payload["max_completion_tokens"] = retry_limit
                elif "max_tokens" in payload:
                    payload["max_tokens"] = retry_limit
                else:
                    payload[token_field] = retry_limit
                payload.setdefault("reasoning_effort", "minimal")
                response = retryable_post_json(url, payload, timeout, headers)
                answer = str(response["choices"][0]["message"]["content"] or "")
            if not answer.strip():
                raise RuntimeError(f"LLM returned an empty answer: {response}")
        debug(f"LLM returned {len(answer)} characters")
        return answer
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected LLM response shape: {response}") from exc


def result_dict(query: str, answer: str, results: List[SearchResult], pages: List[CrawledPage]) -> Dict[str, Any]:
    return {
        "query": query,
        "answer": answer,
        "sources": [{
            "title": page.title,
            "url": page.url,
            "success": page.success,
            "error": page.error
        } for page in pages],
        "search_results": [result.__dict__ for result in results],
        "crawled_pages": [page.__dict__ for page in pages],
    }


def run_web_search(query: str,
                   max_results: Optional[int] = None,
                   max_pages: Optional[int] = None,
                   max_chars_per_page: Optional[int] = None,
                   timeout: Optional[float] = None) -> Dict[str, Any]:
    seconds = timeout if timeout is not None else float(os.environ.get("WEBSEARCH_TIMEOUT_SECONDS", "60"))
    result_count = max_results if max_results is not None else env_int("WEBSEARCH_MAX_RESULTS", 5)
    page_count = max_pages if max_pages is not None else env_int("WEBSEARCH_MAX_PAGES", 3)
    page_chars = max_chars_per_page if max_chars_per_page is not None else env_int("WEBSEARCH_MAX_CHARS_PER_PAGE", 6000)

    debug(f"run query={query!r} timeout={seconds}s")
    search_results = call_serpapi(query, result_count, seconds)
    if not search_results:
        raise RuntimeError("SerpAPI returned no usable organic results")
    pages = asyncio.run(crawl_with_crawl4ai(search_results, query, page_count, page_chars))
    if not any(page.content.strip() for page in pages):
        raise RuntimeError("Crawl4AI did not return any non-empty page content")
    answer = call_llm(query, pages, seconds)
    return result_dict(query, answer, search_results, pages)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one WebSearch query via SerpAPI, Crawl4AI, and vLLM/OpenAI chat.")
    parser.add_argument("--query", required=True, help="Question/query to answer.")
    parser.add_argument("--env-file", help="Optional .env path. Defaults to ./env then judge/.env.")
    parser.add_argument("--max-results", type=int)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--max-chars-per-page", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--text", action="store_true", help="Print only the synthesized answer.")
    parser.add_argument("--debug", action="store_true", help="Print stage progress to stderr.")
    args = parser.parse_args()

    discover_env_file(args.env_file)
    global DEBUG
    DEBUG = args.debug
    try:
        output = run_web_search(args.query, args.max_results, args.max_pages, args.max_chars_per_page, args.timeout)
    except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(json.dumps({"ok": False, "error": str(exc), "query": args.query}, indent=2), file=sys.stderr)
        return 2

    if args.text:
        print(output["answer"])
    else:
        print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
