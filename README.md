# RAG Earnings Analyst

A local RAG pipeline that answers analyst questions from earnings call transcripts using LangChain, FAISS, and a locally-run LLM (Ollama). Now supports any transcript — local file or URL — plus optional live market data via yfinance.

## Why This Exists

FP&A analysts and investment researchers spend hours combing through earnings call transcripts looking for guidance figures, risk disclosures, and management commentary. This tool lets you ask:

- "What did management say about capex guidance for FY2025?"
- "Were there any covenant concerns mentioned?"
- "Summarize the CFO's commentary on free cash flow."
- "How does that guidance compare to where the stock is trading?"

...and get a sourced answer in seconds — grounded in the actual transcript text, with optional live market context for the relevant ticker.

## How It Works

1. A transcript (local `.txt` file or a URL) is loaded and chunked into overlapping passages.
2. Each chunk is embedded using a local sentence-transformer model (no OpenAI key needed).
3. Chunks are indexed in a FAISS vector store for fast similarity search.
4. (Optional) Live market data — price, P/E, market cap, 52-week range — is pulled for a given ticker via yfinance.
5. When you ask a question, the top-k most relevant transcript chunks (plus market data, if provided) are passed as context to a local LLM (Llama 3 via Ollama), which generates a grounded answer.

```
Transcript (file or URL) ──► Chunk ──► Embed ──► FAISS Index
                                                       │
Ticker (optional) ──► yfinance ──► Market snapshot ──┤
                                                       ▼
                              Question ──► Retrieve top-k ──► LLM ──► Answer
```

## Requirements

Python 3.10+

```bash
pip install -r requirements.txt
```

No API keys needed for the language model. Runs fully locally using Ollama.

Install Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull llama3
```

## How to Run

**Demo mode** (uses the included Microsoft Q3 FY2026 sample transcript, runs 4 preset questions):

```bash
python main.py
```

**Any transcript, local file:**

```bash
python main.py --source path/to/transcript.txt
```

**Any transcript, from a URL:**

```bash
python main.py --source https://example.com/earnings-call-transcript.txt
```

**With live market data for a ticker** (adds price, P/E, market cap, 52-week range as context):

```bash
python main.py --source sample_transcript.txt --ticker MSFT
```

**Interactive mode** — ask your own questions instead of the preset demo set:

```bash
python main.py --source sample_transcript.txt --ticker MSFT --interactive
```

**Tune retrieval and chunking** (optional):

```bash
python main.py --source sample_transcript.txt --k 6 --chunk-size 600 --chunk-overlap 75
```

**Use a different local LLM or embedding model**:

```bash
python main.py --llm-model mistral --embedding-model sentence-transformers/all-mpnet-base-v2
```

## Sample Output

```
Q: What did management say about capex guidance?
A: CFO Amy Hood guided $190 billion in capex for calendar year 2026, up 61%
from 2025, including $25 billion from higher component pricing. Q4 capex
expected to exceed $40 billion.

Q: Summarize the CFO commentary on margins.
A: Gross margin came in at 67.6%, the narrowest since 2022 due to data center
depreciation. Q4 operating margin guided at ~44%, down from 46.3% in Q3.

Q: What were the key AI metrics mentioned?
A: AI business ARR surpassed $37 billion, up 123% YoY. M365 Copilot seats
exceeded 20 million. GitHub Copilot active in 140,000 organizations.
```

With `--ticker MSFT`, answers can also reference current price, P/E, and how
guidance compares to where the stock is trading.

## Project Structure

```
rag-earnings-analyst/
├── main.py                # Main RAG pipeline — supports file/URL input + yfinance
├── sample_transcript.txt  # Microsoft Q3 FY2026 earnings call transcript (demo)
├── requirements.txt       # Dependencies
└── README.md
```

## Sample Transcript

The repo includes `sample_transcript.txt` — Microsoft's actual Q3 FY2026 earnings
call (April 29, 2026), sourced from public SEC filings and investor relations.
Key topics: Azure growth, $190B capex guidance, AI business ARR, cloud margins.

## What's Next

- Support for PDF transcripts directly (currently text only)
- Multi-document comparison (e.g. compare guidance across two quarters)
- Source citation in answers (which chunk/section the answer came from)

## About

Built by [Charvee Patel](https://github.com/charveepat) — MS Finance (Data Analytics), UIUC Gies College of Business. This project demonstrates practical application of RAG architecture to financial document analysis, relevant to FP&A automation and AI-assisted investment research.
