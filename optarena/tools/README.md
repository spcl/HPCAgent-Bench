# OptArena Agent Tools

This folder contains a standalone, async FastAPI service for external agents.
It is intentionally separate from the current benchmark internals so the HTTP
contract can stabilize first.

The service exposes simple API tools:

- `POST /web/search` - working SerpAPI Google search.
- `POST /web/research` - OpenAI-compatible query planning, concurrent SerpAPI searches, and optional summarization.
- `POST /web/research/agentic-internet` - second research backend using `AgenticInternet/agentic-internet`.
- `POST /benchmark/task` - skeleton benchmark task endpoint.
- `POST /benchmark/verify` - skeleton correctness endpoint.
- `POST /benchmark/score` - skeleton performance endpoint.
- `POST /benchmark/evaluate` - skeleton combined verify/score endpoint.
- `GET /health` - readiness and configured judge URL.

The benchmark endpoints deliberately return `implemented: false` for now. Their
request/response shape is stable so agents can integrate today, while the real
compiler, runner, and scoring implementation can be connected later.

## Environment

Copy the example file and fill the keys you need:

```sh
cp optarena/tools/.env.example optarena/tools/.env
```

Required for web search:

```sh
SERPAPI_API_KEY=...
```

Required for LLM summarization:

```sh
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

For OpenAI-compatible local or hosted models, keep `OPENAI_API_KEY` set to the
provider token and point `OPENAI_BASE_URL` at the compatible `/v1` API.

Future Anthropic support is represented by:

```sh
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

Optional AgenticInternet backend:

```sh
pip install -r optarena/tools/requirements-agentic-internet.txt
AGENTIC_INTERNET_TIMEOUT_SECONDS=300
```

The AgenticInternet project supports OpenAI, Anthropic, OpenRouter, HuggingFace,
SerpAPI, Exa, and Browser Use keys depending on which tools you ask it to use.
At minimum, configure the LLM provider and search keys required by that package.

Address variables:

```sh
OPTARENA_TOOLS_URL=http://localhost:8801
OPTARENA_JUDGE_URL=http://judge:8800
```

Inside Docker Compose, `http://optarena-tools:8801` and `http://judge:8800`
can be resolved by service name. Outside Docker, use `http://localhost:8801`
for this service and override `OPTARENA_JUDGE_URL` with the reachable judge
address.

## Start Locally

```sh
python3 -m venv .venv-tools
. .venv-tools/bin/activate
pip install -r optarena/tools/requirements.txt
python3 -m optarena.tools.optarena_tools.main
```

To enable the AgenticInternet backend locally:

```sh
pip install -r optarena/tools/requirements-agentic-internet.txt
```

Health check:

```sh
curl http://localhost:8801/health
```

## Start With Docker

```sh
cd optarena/tools
cp .env.example .env
# edit .env
docker compose up --build
```

To include AgenticInternet in the Docker image:

```sh
INSTALL_AGENTIC_INTERNET=true docker compose up --build
```

The API is then reachable from the host at:

```text
http://localhost:8801
```

and from other Compose services at:

```text
http://optarena-tools:8801
```

## Web Search

Simple search:

```sh
curl -X POST http://localhost:8801/web/search \
  -H 'content-type: application/json' \
  -d '{"query":"cache blocking GEMM optimization","num_results":5}'
```

Research with automatic LLM query planning:

```sh
curl -X POST http://localhost:8801/web/research \
  -H 'content-type: application/json' \
  -d '{
    "query": "What are practical GEMM optimization strategies for C?",
    "num_results_per_query": 5,
    "summarize": true
  }'
```

Research with explicit concurrent searches:

```sh
curl -X POST http://localhost:8801/web/research \
  -H 'content-type: application/json' \
  -d '{
    "query": "What are practical GEMM optimization strategies for C?",
    "search_queries": [
      "GEMM cache blocking C optimization",
      "SIMD vectorization matrix multiplication C",
      "OpenBLAS GEMM optimization notes"
    ],
    "num_results_per_query": 5,
    "summarize": true
  }'
```

When `search_queries` is omitted and `OPENAI_API_KEY` is configured, the service
asks the OpenAI-compatible model to plan 2-4 Google queries. The service sends
the resulting SerpAPI requests concurrently with `asyncio.gather`, so multiple
queries in one request do not run serially. FastAPI also handles multiple agent
requests concurrently.

## SerpAPI Web Research Agent Compatibility

The requested reference project is the lightweight `serpapi/web-research-agent`
style flow: plan searches, run SerpAPI calls concurrently, and return a cited
answer. This service implements that API-native behavior directly.

If you later clone that project and want to call its script exactly, configure:

```sh
WEB_RESEARCH_AGENT_PATH=/path/to/web-research-agent/research_agent.py
```

and call:

```json
{
  "query": "What are the latest approaches to retrieval augmented generation?",
  "use_external_agent": true
}
```

on `POST /web/research`.

## AgenticInternet Research Backend

The second research version is intentionally a wrapper around the upstream
`AgenticInternet/agentic-internet` package. It does not reimplement that agent.
Install the optional dependency first:

```sh
pip install -r optarena/tools/requirements-agentic-internet.txt
```

Then call:

```sh
curl -X POST http://localhost:8801/web/research/agentic-internet \
  -H 'content-type: application/json' \
  -d '{
    "query": "What are practical GEMM optimization strategies for C?",
    "depth": "quick",
    "output_format": "json"
  }'
```

Request fields:

- `query`: research topic or task.
- `depth`: `quick`, `moderate`, or `deep`.
- `model`: optional model override passed to AgenticInternet where supported.
- `max_iterations`: optional iteration cap where supported.
- `output_format`: `json`, `markdown`, or `text` for CLI mode.
- `use_cli`: set `true` to call `python -m agentic_internet.cli research`; default `false` imports `ResearchAgent` directly.

This gives us two versions to compare later:

- `/web/research`: lightweight OptArena-native SerpAPI/OpenAI-compatible backend.
- `/web/research/agentic-internet`: upstream AgenticInternet backend.

## Benchmark Skeleton Calls

Fetch a task:

```sh
curl -X POST http://localhost:8801/benchmark/task \
  -H 'content-type: application/json' \
  -d '{"kernel":"gemm","language":"c","preset":"S"}'
```

Evaluate source:

```sh
curl -X POST http://localhost:8801/benchmark/evaluate \
  -H 'content-type: application/json' \
  -d '{
    "kernel": "gemm",
    "language": "c",
    "source": "void gemm_fp64(...) { /* candidate */ }",
    "build": []
  }'
```

For now these return a structured placeholder:

```json
{
  "status": "not_implemented",
  "implemented": false,
  "message": "...",
  "request": {...},
  "next_step": "..."
}
```

Later, the same endpoints should call the real benchmark registry, compiler,
runner, correctness oracle, and scorer.

## OpenAI Tool Calling Example

The file `optarena/tools/examples/openai_tools.py` shows standard OpenAI function tools.
The important part is that each LLM tool is just an HTTP call:

```python
def call_tool(name: str, arguments: dict) -> dict:
    endpoint = {
        "web_research": "/web/research",
        "agentic_internet_research": "/web/research/agentic-internet",
        "benchmark_task": "/benchmark/task",
        "benchmark_evaluate": "/benchmark/evaluate",
    }[name]
    response = httpx.post(f"{TOOLS_URL}{endpoint}", json=arguments, timeout=120.0)
    response.raise_for_status()
    return response.json()
```

Run it with:

```sh
pip install openai httpx
OPENAI_API_KEY=... OPTARENA_TOOLS_URL=http://localhost:8801 \
  python3 optarena/tools/examples/openai_tools.py
```

## LangChain Example

The file `optarena/tools/examples/langchain_tools.py` wraps the same HTTP calls with
LangChain `@tool` functions:

```python
@tool
def web_research(query: str) -> dict:
    return _post("/web/research", {"query": query, "summarize": True})
```

Run it with:

```sh
pip install langchain-core langchain-openai httpx
OPENAI_API_KEY=... OPTARENA_TOOLS_URL=http://localhost:8801 \
  python3 optarena/tools/examples/langchain_tools.py
```

## Agent Integration Checklist

1. Start the tools service locally or with Docker.
2. Export `OPTARENA_TOOLS_URL` in the agent environment.
3. Export `SERPAPI_API_KEY` if the agent may use web search.
4. Export `OPENAI_API_KEY` plus optional `OPENAI_BASE_URL` if the service should summarize research.
5. Export `OPTARENA_JUDGE_URL` if the future benchmark judge is not reachable at `http://judge:8800`.
6. Install `requirements-agentic-internet.txt` if the agent should use the AgenticInternet backend.
7. Register each endpoint as an LLM tool in the agent.
8. Let the agent call these tools as normal JSON HTTP tools.

## Endpoint Contract Summary

`POST /web/search`

```json
{"query":"...","num_results":5,"location":null,"language":"en"}
```

`POST /web/research`

```json
{
  "query": "...",
  "search_queries": ["...", "..."],
  "num_results_per_query": 5,
  "summarize": true,
  "use_external_agent": false
}
```

`POST /web/research/agentic-internet`

```json
{
  "query": "...",
  "depth": "quick",
  "model": null,
  "max_iterations": null,
  "output_format": "json",
  "use_cli": false
}
```

`POST /benchmark/task`

```json
{"kernel":"gemm","language":"c","preset":"S"}
```

`POST /benchmark/evaluate`

```json
{
  "kernel": "gemm",
  "language": "c",
  "source": "...",
  "library": null,
  "build": [],
  "workspace_bytes": null,
  "preset": "S"
}
```
