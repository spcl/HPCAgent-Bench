# Judge Tools

The AMD judge-agent image copies this directory to `/opt/optarena-judge`. It holds one tool,
`web_search`: `experiments/judge_service.py` reads it from `/opt/optarena-judge/tools`, and
`experiments/run_cluster.sh` puts `containers/judge/tools` on the judge's `PYTHONPATH`.

`web_search` is process-oriented and runs once per query:

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
```

The network-free test is `tests/test_judge_web_search.py` at the repository root.

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
WEBSEARCH_LLM_API_KEY=
WEBSEARCH_TIMEOUT_SECONDS=60
WEBSEARCH_MAX_RESULTS=5
WEBSEARCH_MAX_PAGES=3
WEBSEARCH_MAX_CHARS_PER_PAGE=6000
WEBSEARCH_CRAWL_CONCURRENCY=3
WEBSEARCH_CHECK_ROBOTS_TXT=true
WEBSEARCH_PAGE_TIMEOUT_MS=30000
WEBSEARCH_BM25_THRESHOLD=1.0
WEBSEARCH_BM25_LANGUAGE=english
WEBSEARCH_LLM_MAX_TOKENS=4096
WEBSEARCH_LLM_TOKEN_FIELD=max_tokens
WEBSEARCH_LLM_TEMPERATURE=
WEBSEARCH_LLM_REASONING_EFFORT=minimal
WEBSEARCH_LLM_VERBOSITY=low
WEBSEARCH_LLM_EMPTY_RETRY_MULTIPLIER=4
# test/dev only: JSON mapping URL -> page text; when set, Crawl4AI is skipped
WEBSEARCH_FAKE_CRAWL_JSON=
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
