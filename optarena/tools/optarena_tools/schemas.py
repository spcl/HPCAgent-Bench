"""Request and response models for the agent-facing API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    web_search_ready: bool
    llm_ready: bool
    judge_url: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    num_results: int = Field(default=5, ge=1, le=20)
    location: str | None = None
    language: str = "en"


class SearchResult(BaseModel):
    title: str
    link: HttpUrl | str
    snippet: str = ""
    source: str = "serpapi"
    position: int | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    raw: dict[str, Any] | None = None


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1)
    search_queries: list[str] | None = None
    num_results_per_query: int = Field(default=5, ge=1, le=10)
    summarize: bool = True
    use_external_agent: bool = False


class AgenticInternetResearchRequest(BaseModel):
    query: str = Field(min_length=1)
    depth: Literal["quick", "moderate", "deep"] = "quick"
    model: str | None = None
    max_iterations: int | None = Field(default=None, ge=1, le=100)
    output_format: Literal["json", "markdown", "text"] = "json"
    use_cli: bool = False


class Citation(BaseModel):
    title: str
    link: HttpUrl | str
    snippet: str = ""


class ResearchResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    trace: dict[str, Any]


class BenchmarkTaskRequest(BaseModel):
    kernel: str = Field(min_length=1)
    language: str | None = None
    preset: str | None = None


class BenchmarkSubmission(BaseModel):
    kernel: str = Field(min_length=1)
    language: str | None = None
    source: str | None = None
    library: str | None = None
    build: list[str] = Field(default_factory=list)
    workspace_bytes: str | int | None = None
    preset: str | None = None


class SkeletonResponse(BaseModel):
    status: Literal["accepted", "not_implemented"]
    implemented: bool
    message: str
    request: dict[str, Any]
    next_step: str
