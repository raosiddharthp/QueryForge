# QueryForge

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
