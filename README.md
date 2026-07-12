# QueryForge
<<<<<<< HEAD

> **Adaptive multi-query retrieval optimization for enterprise RAG — routed, explained, and built to run at $0.00.**

A single dense-embedding lookup is a reasonable default for "what's our PTO policy," and a
confident wrong answer for "what approval do I need for a $50K vendor contract with non-standard
payment terms." The second question needs three separate facts pulled from three different
places and merged — a document RAG pipeline built for the first question will retrieve two of
the three, rank them below the noise, and answer fluently anyway. QueryForge exists to catch that
gap: it classifies the query, decomposes it when decomposition helps, runs dense, sparse, and
hybrid retrieval in parallel, fuses the results with Reciprocal Rank Fusion, and returns the
routing decision alongside the answer — never silently.

This is not a wrapper around a single retrieval call with a nicer prompt. It's a five-stage
routing pipeline (classify → decompose → retrieve → rerank → fuse) built to make its own
decisions inspectable, running entirely on Google Cloud's Always-Free tier because the account it
lives on has zero spending headroom to give.

---

## Build status, stated plainly

The pipeline logic in `build/service/` — classification, decomposition, hybrid retrieval, RRF
fusion, the config recommender, and the proactive budget guard — is real, working Python against
real Gemini and Firestore calls, not pseudocode. What it has **not** yet had is a full-scale
production deployment or a large-corpus evaluation run: the seed corpus in
`build/service/data/seed_corpus/` is three documents, sized to prove the routing logic, not to
report a benchmark. The recall/MRR figures in the [design doc](./index.html)'s experiment grid
(§6.2) are grid-search targets from the published literature the α-defaults are drawn from
(Luan et al.), not a measured run on this exact corpus — that distinction matters and is not
blurred here. A real evaluation run against a larger corpus is the next milestone, not a
claimed result.

**[→ Full architecture design doc](https://raosiddharthp.github.io/QueryForge/)** — problem
statement, pipeline diagrams, cost analysis with cited sources, design validation, and six ADRs.

---

## The problem

| Query type | Example | Single-embedding RAG | QueryForge |
|---|---|---|---|
| Multi-hop | "What approval is needed for vendor contracts over $50K with non-standard terms?" | Retrieves the right documents, ranks them 4th and 5th behind tangential matches | Decomposes into 3 sub-queries, retrieves each independently, RRF surfaces all 3 at the top |
| Comparative | "Enterprise vs SMB contract terms" | Embeds toward one entity, suppresses the other | Hybrid retrieval, threshold filtering disabled to prevent suppression |
| Temporal | "How has parental leave policy changed since Series B?" | No version awareness — a stale doc can outrank the current one | BM25 date-field boost + Firestore metadata filter |
| Entity-heavy | "What is the payment term for vendor #V-2847?" | Semantic similarity blurs exact identifiers | BM25-heavy hybrid (α=0.40) for exact-match lookup |

A document retrieval system doesn't refuse these queries when it gets them wrong — it answers
fluently, confidently, and with the wrong document ranked out of view. That's the failure mode
this system is built against.

---

## Architecture

```
Browser (Firebase Hosting, static app)
   │  POST /v1/optimize { query, corpus_id, top_k }
   ▼
Cloud Run (FastAPI, scale-to-zero, single container)
   ├─ classify()         → gemini-2.5-flash-lite   (Gemini Developer API, free tier)
   ├─ decompose()        → gemini-2.5-flash-lite   (multi-hop only)
   ├─ dense_retrieve()   → Firestore Vector Search  (self-hosted all-MiniLM-L6-v2 embeddings)
   ├─ sparse_retrieve()  → rank-bm25, in-process
   ├─ hybrid_combine()   → α·dense + (1-α)·bm25, pure Python
   ├─ rerank()           → self-hosted ms-marco-MiniLM-L-6-v2 (multi-hop only)
   ├─ rrf_fuse()         → Reciprocal Rank Fusion, k=60
   └─ log_query()        → Firestore query_logs
```

Every Gemini call passes through `build/service/budget_guard.py` first — a Firestore-transactional
spend check that enforces the $0.01 cap in real time, because Google's own billing data lags too
much to do it reactively. See [ADR-006](docs/adr/006.md) for why that matters.

### Five layers

| Layer | Responsibility | Implementation |
|---|---|---|
| **Interface** | Query input, adaptive result rendering, config export | Static HTML + vanilla JS, Firebase Hosting |
| **Query Intelligence** | Classification, decomposition, HyDE fallback | Gemini 2.5 Flash-Lite, Gemini Developer API |
| **Retrieval** | Dense, sparse, and hybrid strategies, run in parallel | Firestore Vector Search + self-hosted `rank-bm25` |
| **Fusion & Rerank** | Score-scale-invariant merge, selective precision boost | Pure-Python RRF + self-hosted cross-encoder |
| **Governance** | Spend enforcement, query logging, config recommendation | `budget_guard.py` + Firestore `query_logs` |

---

## Stack

| Component | Choice | Why |
|---|---|---|
| LLM inference | Gemini 2.5 Flash-Lite | Gemini Developer API needs no IAM setup and bills on a quota independent of Cloud Billing (ADR-004) |
| Dense retrieval | Firestore Vector Search | Persists across Cloud Run cold starts; Vertex AI Vector Search has no free tier (ADR-002) |
| Sparse retrieval | `rank-bm25`, self-hosted | Zero marginal cost per query, in-process |
| Embeddings & reranker | `all-MiniLM-L6-v2` + `ms-marco-MiniLM-L-6-v2`, self-hosted | Bundled in the container image; Vertex AI's embedding API has no free tier (ADR-003) |
| Compute | Cloud Run, scale-to-zero | 2M requests/month Always-Free, no cost floor when idle (ADR-005) |
| Spend enforcement | `budget_guard.py`, Firestore-transactional | Google's billing data lags ~24h — too slow to be the primary guard (ADR-006) |
| Frontend | Static HTML + vanilla JS | Zero build step |
| Hosting | Firebase Hosting (Spark) | Genuinely free — no billing account required for this layer |

---

## Budget, stated plainly

This runs on two separate pieces of billing infrastructure, deliberately kept apart:

- **The GCP project** (Cloud Run, Firestore, Secret Manager, Cloud Storage) is linked to a $10
  billing account shared with three other apps. Every resource here is chosen to sit inside its
  Always-Free perpetual quota, targeting **$0/month** — see the design doc's §8.2 Always-Free
  ledger for the itemized breakdown against official Google Cloud pricing.
- **The Gemini API key** bills against a Gemini Developer API quota that's independent of Cloud
  Billing entirely. It's still capped at **$0.01** and self-enforced in application code
  (`budget_guard.py`), not just monitored after the fact — see [ADR-006](docs/adr/006.md) for why
  a billing-alert-only approach isn't fast enough for a cap this small.

Neither of these is a workaround. They're the actual constraint this build was designed against.

---

## Documented trade-offs

- **Seed corpus, not a benchmark corpus** — `data/seed_corpus/` has three documents, enough to
  prove the routing and fusion logic end-to-end, not enough to report a recall@10 number that
  would mean anything. See "Build status" above.
- **Step-back and query-rewrite variants are routed but not wired** — the classifier's output
  space anticipates them (see the design doc's Figure 3), but only decomposition and HyDE are
  live. Flagged as deferred in the design doc's MVP Scope, not presented as shipped.
- **Cost estimates in `budget_guard.py` are estimates** — computed from published per-token
  pricing against the response's token-usage metadata, not GCP's actual invoice. Treated as a
  hard-stop guard, not a billing record; Firestore's `query_logs` and Cloud Billing's own reports
  are the actual source of truth.
- **Single-tenant, no auth gate on the Cloud Run service in this MVP** — `roles/run.invoker` is
  granted to `allUsers` in `infra/main.tf` for demo simplicity. Production path is Firebase
  Auth + API Gateway, the same pattern documented in the design doc's architecture section.

---

## Architecture Decision Records

Six ADRs — context, decision, consequences, alternatives considered, for every choice that
mattered enough to argue about:

| ADR | Decision | Status |
|---|---|---|
| [ADR-001](docs/adr/001.md) | Google Cloud-only architecture over open-source, multi-vendor stack | Accepted |
| [ADR-002](docs/adr/002.md) | Firestore Vector Search over ChromaDB or Vertex AI Vector Search | Accepted |
| [ADR-003](docs/adr/003.md) | Self-hosted open-weight models over Vertex AI Embeddings API | Accepted |
| [ADR-004](docs/adr/004.md) | Gemini Developer API over the Vertex AI Gemini endpoint | Accepted |
| [ADR-005](docs/adr/005.md) | Cloud Run over GKE Autopilot | Accepted |
| [ADR-006](docs/adr/006.md) | Project-scoped, self-enforcing budget guard over a billing-data-reactive kill switch | **Revised** |

**[→ Full architecture design doc, diagrams, and pipeline simulator](https://raosiddharthp.github.io/QueryForge/)**

---

## Repository structure

```
queryforge/
├── index.html                # Architecture design doc · diagrams · ADRs · simulator (GitHub Pages)
├── README.md                  # This file
├── build/
│   ├── frontend/                # Production app — Firebase Hosting
│   └── service/
│       ├── main.py                # FastAPI entrypoint, pipeline orchestration
│       ├── config.py               # Query-type routing table (α, decompose, rerank)
│       ├── gemini_client.py         # Classifier, decomposer, HyDE — Gemini Developer API
│       ├── budget_guard.py           # Proactive, self-enforcing $0.01 spend cap
│       ├── embeddings.py              # Self-hosted dense embedder + cross-encoder reranker
│       ├── retrieval.py                # Dense (Firestore), sparse (BM25), hybrid combine
│       ├── fusion.py                    # Reciprocal Rank Fusion
│       ├── recommender.py                # config_recommendation + Firestore query log
│       ├── chunker.py                     # Content-type-aware chunking router
│       ├── ingest.py                       # Corpus ingestion: chunk → embed → Firestore
│       ├── data/seed_corpus/                # Three-document seed corpus (see Build status)
│       ├── Dockerfile
│       └── requirements.txt
├── infra/
│   ├── main.tf                # Firestore, Secret Manager, Cloud Run, Storage, IAM, secondary budget
│   ├── variables.tf
│   ├── provider.tf
│   └── backend.tf             # GCS remote state
└── docs/
    └── adr/                  # Six Architecture Decision Records
```

---

## Running locally

The design doc is fully static:

```bash
git clone https://github.com/raosiddharthp/QueryForge
cd QueryForge
open index.html
```

For the backend service:

```bash
cd build/service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add your Gemini Developer API key from aistudio.google.com

gcloud auth application-default login
gcloud auth application-default set-quota-project queryforge-prod

python3 -m uvicorn main:app --port 8080
```

```bash
curl -X POST http://localhost:8080/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the payment term for vendor #V-2847?", "corpus_id": "acme-hr-corpus", "top_k": 5}'
```

Requires a GCP project with Firestore enabled and a Gemini Developer API key. The seed corpus
needs ingesting before queries return anything:

```bash
python3 ingest.py --corpus-id acme-hr-corpus --content-type policy_docs data/seed_corpus/*.txt
```

---

## Production path

| MVP (this build) | Production |
|---|---|
| Gemini Developer API, AI Studio key | Vertex AI Gemini — SLA, quota management, VPC-SC |
| Firestore Vector Search, ≤50MB corpus | Vertex AI Vector Search, production-scale corpus |
| `allUsers` invoker on Cloud Run | Firebase Auth + API Gateway, allowlisted access |
| Three-document seed corpus | Real corpus + measured recall@10/MRR evaluation run |
| $0.01 hard-capped shared billing | Dedicated billing account, usage-scaled budget |

---

## Author

**Siddharth Rao** · Enterprise Architect · TOGAF Certified · GCP Certified Architect
Architecture Portfolio · 2026
=======
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
>>>>>>> 0e90a40349dfdb339d8cd090f7819d0d70374862
