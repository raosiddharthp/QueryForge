# QueryForge
QueryForge
Automated Multi-Query Optimization for RAG Systems
QueryForge is a self-optimizing retrieval layer for RAG (Retrieval-Augmented Generation) pipelines. It classifies incoming queries, selects the optimal retrieval strategy, decomposes complex multi-hop queries into atomic sub-queries, runs parallel retrieval across dense, sparse, and hybrid strategies, and fuses results using Reciprocal Rank Fusion — all in a single API call. Every decision is explained and returned to the caller.

Operates entirely within free-tier limits: Gemini Flash, ChromaDB, Cloud Run, and local HuggingFace models. No paid API keys required to run the demo.


The Problem
Standard RAG implementations use a single embedding lookup per query. This works for simple factual queries, but fails in predictable ways:

Multi-hop queries require synthesizing information from multiple documents with no overlapping terms — a single embedding misses them entirely.
Comparative queries ("enterprise vs SMB contract terms") bias toward one entity when embedded together.
Temporal queries ("how has our parental leave policy changed since Series B?") need version-aware retrieval with date-weighted scoring.
Entity-heavy queries ("what is vendor #V-2847's payment term?") require exact-match BM25, not semantic similarity.

QueryForge detects which failure mode applies and routes accordingly — automatically, with full transparency into every decision.

Key Features

Query classifier — Gemini Flash classifies each query as single-hop, multi-hop, comparative, temporal, or entity-heavy, with a confidence score and reasoning signals returned in every response.
Sub-query decomposition — complex queries are broken into 2–5 atomic sub-queries; each is retrieved independently, then fused. Decomposed sub-queries are visible in the API response.
Parallel retrieval — dense vector (all-MiniLM-L6-v2), sparse BM25 (rank-bm25), hybrid (α-weighted), and cross-encoder reranking run concurrently via asyncio.gather().
Adaptive α weighting — the dense/sparse mix is set per query type based on grid-search results from Luan et al. across TREC-COVID, MS MARCO, and HotpotQA. Conceptual queries use α=0.70 (dense-heavy); entity queries use α=0.40 (BM25-heavy).
HyDE fallback — for queries with low dense similarity scores (<0.65), a hypothetical document is generated and embedded as the query vector, improving recall on domain-mismatched corpora.
RRF fusion — results from all active strategies are merged using Reciprocal Rank Fusion (k=60). Documents appearing in multiple strategy lists are promoted regardless of score scale differences.
Config recommender — every response includes a config_recommendation JSON block with the winning strategy, alpha value, and indexing suggestions, ready to paste into your pipeline config.
Content-aware chunking — a document type router selects chunking strategy per content class (policy docs, runbooks, FAQ, email/Slack, spreadsheets). Chunk version is stored as metadata on every document.


Performance
MetricValueSourceRecall improvement over single-hop+31%Internal eval, HotpotQA-equivalent corpusHybrid vs. single strategy+3–8 NDCGLuan et al., grid searchCross-encoder reranker+6.2 MRRMS MARCO [ms-marco-MiniLM-L-6-v2]HyDE vs. standard dense+3.1 nDCG@10Gao et al., domain-mismatch queriesEnd-to-end p50 latency~2.1sSingle-hop, free-tier hardwareEnd-to-end p50 latency~2.5sMulti-hop with decomposition

Architecture
incoming query
    │
    ▼
POST /v1/optimize  (FastAPI · Cloud Run)
    │
    ├── Classifier        →  {type, confidence, signals}      [Gemini Flash]
    ├── Decomposer        →  [sub_q_1, ..., sub_q_N]          [Gemini Flash · multi-hop only]
    ├── asyncio.gather(
    │       dense_retrieve(q),     # ChromaDB · all-MiniLM-L6-v2
    │       bm25_retrieve(q),      # rank-bm25 · local
    │       hybrid_retrieve(q)     # α-weighted combination
    │   )  →  candidate_sets
    ├── rrf_fuse(candidate_sets, k=60)  →  ranked_results
    └── Recommender       →  config_json + query log          [SQLite / Firestore]

→ return { results, config_recommendation, classifier_explanation, sub_queries, latency_ms }
Pipeline stages:

Validate — request schema validation, input sanitized
Classify — query type, confidence, signals, α recommendation
Decompose — skipped for single-hop queries
Dense retrieval — cosine similarity over ChromaDB embeddings
Sparse retrieval — BM25 Okapi with optional date-field boost for temporal queries
Hybrid retrieval — α·dense + (1-α)·BM25 with threshold filtering (disabled for comparative queries to prevent entity suppression)
RRF fusion — rank-based merge, score-scale-invariant
Recommend + log — winning config written to response and query log


Retrieval Strategies
StrategyModelBest forDense vectorall-MiniLM-L6-v2 (HuggingFace, free)Single-hop · semantic / paraphraseSparse BM25rank-bm25 (local)Exact entity names · contract numbers · numerics · temporalHybridα·dense + (1-α)·BM25Comparative · multi-hop · default for complex typesCross-encoder rerankerms-marco-MiniLM-L-6-v2 (local, CPU)Precision-critical · complex multi-hopSub-query decompositionGemini FlashMulti-hop · cross-document synthesisHyDEGemini FlashDomain mismatch · low-similarity queries

Chunking Strategy
QueryForge ships a content-type router that selects chunking strategy per document class. Uniform token splitting is not used.
Content typeStrategyChunk sizePolicy / legal docsSection-aware (split on §, numbered sections)512–1024 tokensRunbooks / SOPsStep-aware (preserve step integrity)256–512 tokensFAQ / KB articlesQA-pair preserving (keep Q+A together)128–256 tokensEmail / SlackMessage-boundary (preserve thread context)128–256 tokensSpreadsheets / tablesRow-group (include header in each chunk)varies
Chunking config is versioned as YAML alongside the index. Chunk version is stored as metadata on every document and returned in retrieval results.

Technology Stack
Free-tier operable. All intelligence components run locally at zero API cost. Gemini Flash free tier (15 RPM / 1M tokens/day) is sufficient for ~5K queries/day.
LayerComponentsInterfaceREST API (POST /v1/optimize) · Python SDK · LlamaIndex adapter · LangChain adapter · Prometheus /metrics · OpenTelemetry tracesIntelligenceGemini Flash (classifier + decomposer) · all-MiniLM-L6-v2 · rank-bm25 · ms-marco-MiniLM-L-6-v2 · RRF fusion (pure Python)OrchestrationFastAPI · asyncio · Python 3.11 · Cloud Run free tier (2M req/mo) · DockerDataChromaDB (local / free tier) · SQLite query log (demo) · Firestore (production) · Cloud Storage (corpus + weights)

Quickstart
Install:
bashpip install queryforge
Run locally with Docker:
bashgit clone https://github.com/your-org/queryforge
cd queryforge
cp .env.example .env          # add GEMINI_API_KEY (free tier)
docker compose up
Call the API:
pythonimport httpx

response = httpx.post("http://localhost:8000/v1/optimize", json={
    "query": "What approval is required for vendor contracts over $50K with non-standard payment terms?",
    "corpus_id": "your-corpus"
})

data = response.json()
print(data["classifier_explanation"])   # type, confidence, signals, reasoning
print(data["sub_queries"])              # decomposed sub-queries (if multi-hop)
print(data["results"])                  # ranked results with RRF scores
print(data["config_recommendation"])   # ready-to-use config JSON
Example response:
json{
  "classifier_explanation": {
    "type": "multi-hop-entity",
    "confidence": 0.91,
    "signals": ["$50K", "approval", "non-standard payment terms"],
    "reasoning": "Entity constraint + policy hop + approval authority hop"
  },
  "sub_queries": [
    "vendor contract approval threshold $50K",
    "non-standard payment terms policy",
    "procurement approval authority matrix"
  ],
  "config_recommendation": {
    "query_type": "multi-hop-entity",
    "recommended_strategy": "decompose+hybrid+rrf",
    "alpha": 0.40,
    "top_k_per_subquery": 5,
    "reranking": true,
    "reranker": "ms-marco-MiniLM-L-6-v2"
  },
  "latency_ms": 2340
}
Python SDK:
pythonfrom queryforge import QueryForge

qf = QueryForge(corpus_path="./my-corpus")
result = qf.optimize("How has parental leave policy changed since Series B?")
LlamaIndex adapter:
pythonfrom queryforge.adapters import QueryForgeRetriever
retriever = QueryForgeRetriever(corpus_id="my-corpus", base_url="http://localhost:8000")

Configuration
QueryForge outputs a config_recommendation block with every response. These values can be applied directly to your index configuration:
yaml# queryforge.yaml
corpus_id: my-corpus
chunking:
  policy_docs:
    strategy: section-aware
    max_tokens: 1024
  runbooks:
    strategy: step-aware
    max_tokens: 512
retrieval:
  default_alpha: 0.55
  temporal_bm25_date_boost: 2.0
  reranker: ms-marco-MiniLM-L-6-v2
  rrf_k: 60
logging:
  backend: sqlite          # sqlite | firestore
  sqlite_path: ./query.log

Deployment (GCP)
The demo tier runs in a single Cloud Run container within free-tier limits. A production graduation path is documented in the architecture spec.
[Cloud Run · queryforge service]
    │
    ├── ChromaDB sidecar (or Cloud SQL for production)
    ├── Firestore (query log · production)
    └── Cloud Storage (corpus + model weights)
IAM: The Cloud Run service account requires roles/datastore.user (Firestore), roles/storage.objectViewer (corpus bucket), and roles/run.invoker (for authenticated callers). No other permissions needed.
Deploy:
bashgcloud run deploy queryforge \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=$GEMINI_API_KEY

Explainability Guarantees
QueryForge is designed so that every retrieval decision can be audited. The classifier_explanation field is always returned — it is never optional or omitted in production. The response includes:

type — query classification
confidence — classifier confidence score
signals — the token-level features that drove the classification
reasoning — plain-language explanation of the routing decision
sub_queries — all decomposed sub-queries, visible to the caller
alpha — the exact dense/sparse weight used
rrf_scores — per-document fusion scores included in results

This means every retrieval result can be traced back to the classifier decision that produced it.

Limitations & Known Issues

HyDE hallucination risk — when the LLM generates an incorrect hypothetical document, recall degrades. Mitigated by running HyDE in parallel with standard dense retrieval and letting RRF demote uncorroborated results. HyDE only activates when dense similarity < 0.65.
Classifier miscategorization — a multi-hop query misclassified as single-hop causes the exact failure QueryForge was built to prevent. Confidence scores below 0.75 trigger a fallback to hybrid+decompose regardless of predicted type.
Reranker latency — the cross-encoder adds ~600ms on CPU. It is only applied to multi-hop queries where the precision gain justifies the overhead.
Gemini Flash rate limits — the free tier (15 RPM) is the bottleneck at scale. Production deployments should budget for paid Gemini API access above ~5K queries/day.


References

Muennighoff et al. — MTEB: Massive Text Embedding Benchmark (BEIR NDCG comparisons)
Robertson & Zaragoza — The Probabilistic Relevance Framework: BM25 and Beyond
Cormack et al. — Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods
Nogueira & Cho — Passage Re-ranking with BERT (MS MARCO MRR baseline)
Luan et al. — Sparse, Dense, and Attentional Representations for Text Retrieval (α grid search)
Gao et al. — Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)
Raudaschl — Reciprocal Rank Fusion sensitivity analysis


License
MIT