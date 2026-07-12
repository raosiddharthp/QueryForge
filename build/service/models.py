from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    corpus_id: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class ClassifierExplanation(BaseModel):
    type: str
    confidence: float
    signals: list[str]
    reasoning: str
    fallback_applied: bool


class RankedResult(BaseModel):
    doc_id: str
    text: str
    rrf_score: float
    source_strategies: list[str]


class ConfigRecommendation(BaseModel):
    query_type: str
    recommended_strategy: str
    alpha: float
    top_k_per_subquery: int
    reranking: bool
    reranker: str | None = None


class OptimizeResponse(BaseModel):
    results: list[RankedResult]
    config_recommendation: ConfigRecommendation
    classifier_explanation: ClassifierExplanation
    sub_queries: list[str]
    hyde_triggered: bool
    latency_ms: int
