# RAG Earnings Analyst

Ask investment questions. Get answers grounded in actual earnings call transcripts — not LLM training data guesses.

```
Q: What did the CFO say about margin guidance?

A: [Microsoft (MSFT) — Q2 FY2024, CFO Amy Hood]
   Operating margin expanded to 44%, up 500 basis points year over year.
   Capital expenditures were $11.5B, driven by AI and cloud infrastructure.
```

## Why this exists

FP&A analysts and investment researchers spend hours combing through earnings transcripts for guidance figures, risk disclosures, and management commentary. This tool lets you query any transcript in plain English and get answers cited back to the source — company, quarter, and speaker.

The core idea: **retrieval quality matters more than model quality.** Getting the right 4 chunks into context beats upgrading to a bigger model on bad context. This project tests that hypothesis across two retrieval architectures.

## How it works

```
INDEX (one-time)
  Transcript → Chunk (500 words, 50 overlap) → Embed → FAISS vector store

RETRIEVE (per question)
  Question → Embed → Cosine similarity → Top-k chunks

GENERATE
  [Chunks with source headers] + Question → LLM → Cited answer
```

Each retrieved chunk is labeled with its source before being sent to the LLM:
```
[NVIDIA (NVDA) — Q4 FY2024, CEO Jensen Huang]
Data center revenue reached $18.4 billion, up 409% year over year...
```

This forces the model to cite who said what rather than blending sources.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# open .env and paste your ANTHROPIC_API_KEY (needed for advanced mode)
```

Basic mode also requires [Ollama](https://ollama.ai) running locally:
```bash
ollama pull llama3
```

## Quickstart

```bash
# Built-in Big Tech sample — works immediately, no internet needed
python main.py --sample --demo

# Analyze any company (auto-fetches transcript from Motley Fool)
python main.py

# Advanced mode: 5 RAG techniques + Claude generation
python main.py --sample --mode advanced --demo --eval

# Your own transcript file
python main.py --source path/to/transcript.txt --ticker MSFT
```

## Two modes

### Basic (default)
- LangChain + FAISS for vector storage
- Ollama (llama3) as the local LLM — no API costs
- Pure dense retrieval (cosine similarity only)

### Advanced (`--mode advanced`)
Five techniques stacked on top of the basic pipeline, using Claude for generation:

| # | Technique | What it does |
|---|---|---|
| T1 | Hybrid Search | BM25 keyword + dense embeddings — tunable with `--alpha` |
| T2 | HyDE | Claude writes a hypothetical ideal answer, embeds *that*, then searches |
| T3 | Re-ranking | CrossEncoder reads (query, passage) together for precise relevance |
| T4 | Query Decomposition | Breaks complex questions into 2-4 atomic sub-queries, each retrieved independently |
| T5 | RAG Evaluation | RAGAS-style scoring: faithfulness / relevance / precision / recall |

Full pipeline per question: decompose → hybrid search per sub-query → merge → rerank → HyDE verify → rerank → generate → evaluate.

## Key flags

| Flag | Default | Description |
|---|---|---|
| `--sample` | off | Built-in NVDA/MSFT/AAPL/META/AMZN dataset — no fetch needed |
| `--mode` | basic | `basic` or `advanced` |
| `--demo` | off | Run preset questions; interactive if off |
| `--eval` | off | Print RAGAS scores after each answer (advanced only) |
| `--alpha` | 0.5 | Hybrid search: 0.0 = pure BM25, 1.0 = pure dense |
| `--k` | 4 | Chunks retrieved per question |
| `--ticker` | prompt | Stock ticker for live price/P/E/market cap context (yfinance) |
| `--chunk-size` | 500 | Characters per chunk |
| `--llm-model` | llama3 | Ollama model (basic mode only) |
| `--anthropic-model` | claude-haiku-4-5 | Claude model (advanced mode only) |

## Project structure

```
rag-earnings-analyst/
├── main.py               — full RAG pipeline (basic + advanced modes)
├── sample_transcript.txt — Microsoft Q3 FY2026 transcript (demo for live-fetch mode)
├── requirements.txt      — all dependencies
├── .env.example          — API key template
├── .gitignore
├── DEVLOG.md             — build log: what was tried, what broke, what changed
└── README.md
```

## What's in the built-in sample

10 excerpts from Big Tech earnings calls (public transcripts), each tagged with company, ticker, quarter, and speaker:

| Company | Ticker | Quarter | Speakers |
|---|---|---|---|
| NVIDIA | NVDA | Q4 FY2024 | CEO Jensen Huang, CFO Colette Kress |
| Microsoft | MSFT | Q2 FY2024 | CEO Satya Nadella, CFO Amy Hood |
| Apple | AAPL | Q1 FY2024 | CEO Tim Cook, CFO Luca Maestri |
| Meta | META | Q4 2023 | CEO Mark Zuckerberg, CFO Susan Li |
| Amazon | AMZN | Q4 2023 | CEO Andy Jassy, CFO Brian Olsavsky |

Good cross-company questions to try:
- "Which companies showed the strongest revenue growth?"
- "What are the AI infrastructure investment trends? Who is spending most?"
- "Compare operating margins — who is most profitable?"

## Stack

`sentence-transformers` · `faiss-cpu` · `rank-bm25` · `anthropic` · `langchain` · `yfinance` · `beautifulsoup4`

---

Built by [Charvee Patel](https://github.com/charveepat) — MS Finance (Data Analytics), UIUC Gies.
RAG architecture applied to financial document analysis — relevant to FP&A automation and AI-assisted investment research.
