"""AgenticInternet research backend.

This intentionally uses the upstream ``AgenticInternet/agentic-internet``
package instead of reimplementing its internal agent behavior.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from optarena.tools.optarena_tools.config import settings
from optarena.tools.optarena_tools.schemas import (
    AgenticInternetResearchRequest,
    ResearchResponse,
)
from optarena.tools.optarena_tools.web_search import ToolConfigurationError


def _stringify_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("findings", "answer", "summary", "result", "content"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(result, indent=2, default=str)
    return str(result)


def _package_research(request: AgenticInternetResearchRequest) -> ResearchResponse:
    try:
        from agentic_internet import ResearchAgent
    except ImportError as exc:
        raise ToolConfigurationError(
            "agentic_internet is not installed. Install optarena/tools/requirements-agentic-internet.txt."
        ) from exc

    kwargs: dict[str, Any] = {}
    if request.model:
        kwargs["model"] = request.model
    if request.max_iterations is not None:
        kwargs["max_iterations"] = request.max_iterations

    try:
        researcher = ResearchAgent(**kwargs)
    except TypeError:
        researcher = ResearchAgent()

    result = researcher.research(topic=request.query, depth=request.depth)
    raw_result = None
    if isinstance(result, dict):
        raw_result = json.loads(json.dumps(result, default=str))

    return ResearchResponse(
        query=request.query,
        answer=_stringify_result(result),
        citations=[],
        trace={
            "mode": "agentic_internet_package",
            "depth": request.depth,
            "model": request.model,
            "raw_result": raw_result,
        },
    )


async def _cli_research(request: AgenticInternetResearchRequest) -> ResearchResponse:
    cmd = [
        sys.executable,
        "-m",
        "agentic_internet.cli",
        "research",
        request.query,
        "--depth",
        request.depth,
        "--format",
        request.output_format,
    ]
    if request.model:
        cmd.extend(["--model", request.model])
    if request.max_iterations is not None:
        cmd.extend(["--max-iterations", str(request.max_iterations)])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=settings.agentic_internet_timeout_seconds)
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.communicate()
        raise TimeoutError("agentic-internet research timed out") from exc

    out = stdout.decode("utf-8", errors="replace").strip()
    err = stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"agentic-internet CLI failed with {proc.returncode}: {err}")

    trace: dict[str, Any] = {
        "mode": "agentic_internet_cli",
        "depth": request.depth,
        "model": request.model,
        "returncode": proc.returncode,
        "stderr": err,
    }
    if request.output_format == "json":
        try:
            trace["raw_result"] = json.loads(out)
        except json.JSONDecodeError:
            trace["stdout"] = out
    else:
        trace["stdout"] = out

    return ResearchResponse(query=request.query, answer=out, citations=[], trace=trace)


async def agentic_internet_research(request: AgenticInternetResearchRequest) -> ResearchResponse:
    if request.use_cli:
        return await _cli_research(request)
    return await asyncio.to_thread(_package_research, request)
