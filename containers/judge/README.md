# Judge Placeholder Tools

The judge container is intentionally empty for now. This directory only prepares
the execution part of one future judge tool: `web_search`.

`web_search` is process-oriented. A later service can start it once per query:

```bash
python3 /opt/optarena-judge/tools/web_search.py --query "best rocBLAS batched GEMM API"
```

It reads configuration from `.env` or environment variables, calls SerpAPI for
candidate results, uses Crawl4AI to crawl and query-filter those pages, then asks
an OpenAI/vLLM-compatible chat endpoint to synthesize an answer with sources.

## Files

```text
judge/
  .env.example
  requirements.txt
  tools/web_search.py
  tests/test_web_search.py
```

## Configuration

```bash
cp /opt/optarena-judge/.env.example .env
```

Required:

```bash
SERPAPI_API_KEY=<serpapi-key>
WEBSEARCH_LLM_BASE_URL=http://<vllm-host>:8000/v1
WEBSEARCH_LLM_MODEL=<model-name>
```

Optional:

```bash
SERPAPI_URL=https://serpapi.com/search.json
WEBSEARCH_TIMEOUT_SECONDS=60
WEBSEARCH_MAX_RESULTS=5
WEBSEARCH_MAX_PAGES=3
WEBSEARCH_MAX_CHARS_PER_PAGE=6000
WEBSEARCH_CRAWL_CONCURRENCY=3
WEBSEARCH_CHECK_ROBOTS_TXT=true
WEBSEARCH_BM25_THRESHOLD=1.0
WEBSEARCH_LLM_MAX_TOKENS=4096
WEBSEARCH_LLM_TOKEN_FIELD=max_tokens
WEBSEARCH_LLM_TEMPERATURE=
WEBSEARCH_LLM_REASONING_EFFORT=minimal
WEBSEARCH_LLM_VERBOSITY=low
WEBSEARCH_LLM_EMPTY_RETRY_MULTIPLIER=4
```

## Run

```bash
python3 tools/web_search.py --query "CUDA cooperative groups grid sync examples"
```

JSON output includes:

- `query`
- `answer`
- `sources`
- `search_results`
- `crawled_pages`

## Install Notes

The tool uses Crawl4AI in production. It is not just a raw page fetcher: live mode
uses `arun_many()` for multi-URL crawling, `DefaultMarkdownGenerator` with citations,
and `BM25ContentFilter(user_query=<query>)` so each page is reduced to content that
matches the question before it is sent to the LLM.

Install:

```bash
python3 -m pip install -r requirements.txt
playwright install chromium
```
