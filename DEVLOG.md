# DEVLOG — RAG Earnings Analyst

A running log of what was built, what broke, what changed, and why. This is the honest record of how the project evolved across two curriculum sessions.

---

## Session 3 — Initial build (June 17, 2026)

**Goal:** Build a working RAG pipeline over earnings call transcripts. Understand the three-step architecture (index → retrieve → generate) well enough to explain it to a recruiter or engineer.

### Stack chosen
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings — free, local, no API key
- `ChromaDB` for vector storage — simpler API than FAISS for a first build
- Claude API (`claude-opus-4-8`) for generation
- Adaptive thinking enabled (`thinking={"type": "adaptive"}`) — lets the model reason through ambiguous financial questions before answering

### What worked immediately
- The core three-step pipeline ran on the first try: embed → cosine search → Claude answer
- Adaptive thinking visibly improved answer quality on multi-part questions like "compare margin trends across the portfolio" — the model reasoned about which chunks were contradictory before synthesizing
- The source-attribution pattern (`[Company — Quarter, Speaker]` as a header before each chunk) immediately made answers more citable. Without it, Claude blended NVIDIA and Microsoft numbers without attribution.

### What didn't work / what I learned
- **Context order matters.** When I put the highest-relevance chunk last in the context block, answer quality dropped noticeably. LLMs attend more strongly to information at the start and end of long contexts (the "lost in the middle" problem). Fixed: sort retrieved chunks by relevance score, put the best one first.
- **Chunk size was too large initially (1,000 words).** Retrieval pulled in chunks that contained the right paragraph buried inside irrelevant content. Dropped to 500 words with 50-word overlap — precision improved significantly.
- **`response.content[-1].text` vs `response.content[0].text`:** Adaptive thinking inserts a thinking block before the text response. `content[0]` returned the raw chain-of-thought, not the actual answer. Using `content[-1].text` (last block) correctly returns the final text response.
- **ChromaDB in-memory client resets on every run.** The sample data had to be re-indexed each time. Not a problem for a demo, but noted for production: use `PersistentClient(path="./db")`.

### Key insight from this session
RAG is not primarily an LLM problem — it's a retrieval problem. The model is only as good as the chunks you give it. I spent more time tuning chunk size and source labeling than on anything Claude-related.

---

## Between sessions — extended to live transcripts

**Problem:** The Session 3 version only worked on a hardcoded sample dataset. Not useful for real analysis.

**Decision:** Rewrote the input layer to:
1. Auto-search DuckDuckGo for the latest Motley Fool transcript for any company
2. Scrape and extract the article body with BeautifulSoup
3. Fall back to a user-provided URL if scraping fails

Switched from ChromaDB to **FAISS** (via LangChain) because FAISS handles larger documents more efficiently and integrates cleanly with LangChain's document pipeline. ChromaDB was fine for the 10-excerpt demo; FAISS scales to a full 40-page transcript.

Added **yfinance** for live market context — price, P/E, 52-week range injected into the prompt. The idea: grounding the LLM's answer in current market data alongside transcript data gives more actionable output ("guidance was $X, stock is currently trading at $Y which implies a P/E of Z").

Switched from Claude to **Ollama (llama3)** as the default LLM so the tool runs entirely locally — no API costs for basic use.

### What broke
- **Motley Fool scraping is fragile.** The article body selector (`div.article-body`) works ~70% of the time. Some transcripts use `article` or a data attribute. Added fallback chain: try `article-body` div → try `article` tag → try all `<p>` tags.
- **DuckDuckGo blocks automated requests intermittently.** Added `User-Agent` header spoofing. Still fails occasionally — surfaced the URL prompt as a fallback.
- **Large transcripts (40+ pages) took 45+ seconds to embed** on first run because sentence-transformers downloads the model weights. Added a `"this takes ~30 seconds on first run"` log message so the tool doesn't appear frozen.

---

## Session 10 — Advanced RAG techniques (added later)

**Goal:** Understand where naive dense retrieval fails and build the five techniques that fix it.

### The five techniques and why each one exists

**T1 — Hybrid Search (BM25 + dense)**
Naive dense retrieval fails on exact financial terms. A query for "NRR 128%" retrieves chunks about "customer retention" (semantically close) but misses a chunk that literally says "NRR of 128%" (keyword exact match, semantically distant). BM25 catches keyword matches that dense embeddings miss. Combining them with a tunable alpha weights the tradeoff.

*What I found:* alpha=0.3 (70% BM25, 30% dense) worked best for specific metric queries. alpha=0.7 (70% dense) worked better for abstract questions like "what are the risks to this business." Made alpha a CLI flag so users can tune per query type.

**T2 — HyDE (Hypothetical Document Embeddings)**
Dense retrieval embeds the *question* and finds chunks close to it. But a question ("what was NVDA's data center revenue?") lives in a different semantic space than an answer ("data center revenue reached $18.4 billion"). HyDE generates a hypothetical ideal answer, embeds *that*, and searches. The hypothetical document is in the same semantic space as the real answer chunks.

*What I found:* HyDE helped most on abstract questions ("what signals indicate a company faces competitive pressure?"). On specific metric queries, standard retrieval was already good enough. Used HyDE as a *verification step* — run standard retrieval first, then HyDE to catch anything missed.

**T3 — Re-ranking**
Embedding similarity is a blunt instrument — it compares vectors independently. A CrossEncoder reads the query and the passage *together* (full attention across both), which gives much more precise relevance scores. The cost: it's ~10x slower than cosine similarity. Solution: use cosine similarity to get a pool of 10-15 candidates fast, then CrossEncoder to pick the best 4.

*What broke:* CrossEncoder scoring ~100 chunks (a full transcript) was too slow for interactive use (~8 seconds). Fixed by only re-ranking the top 10 candidates from hybrid search, not the full corpus.

**T4 — Query Decomposition**
"Compare NVIDIA's gross margin trajectory to Microsoft's and explain why they're diverging" is actually 3 separate retrieval problems. One query embedding can't optimally retrieve chunks for all three sub-questions simultaneously. Decompose it first, retrieve independently per sub-query, merge the results, then re-rank.

*What I found:* Claude's decomposition is occasionally over-specific ("what is NVIDIA's Q4 FY2024 gross margin" vs "what is NVIDIA's gross margin trend"). Added a fallback: if decomposition returns only 1 sub-query, skip the decompose step and run the original query directly.

**T5 — RAG Evaluation (RAGAS-style)**
Added after noticing that I couldn't tell which technique combination was actually working better. RAGAS gives four scores:
- *Faithfulness*: does the answer add claims not in the retrieved context? (detects hallucination)
- *Answer relevance*: does the answer address the question asked?
- *Context precision*: are the retrieved chunks actually relevant to the question?
- *Context recall*: did retrieval surface everything needed to answer fully?

*What I found:* Context precision was consistently high (above 0.85). Context recall was the weak link — sometimes the right chunk existed in the corpus but retrieval didn't surface it. This confirmed that retrieval, not generation, is the bottleneck. HyDE + hybrid search together improved recall@k by a visible margin over dense-only.

### Architecture decision: class vs functions
Wrapped all five techniques in an `AdvancedRAG` class rather than standalone functions. Reason: the encoder, cross-encoder, BM25 index, and embeddings matrix all need to be initialized once and shared across calls. Loading the CrossEncoder model on every question would add 3-4 seconds per query.

---

## What I would change if building this for production

1. **Persistent index** — FAISS index should serialize to disk so re-indexing doesn't happen on every run. Current code re-embeds every time.
2. **Streaming** — Claude responses stream token-by-token; the current code waits for the full response. For a web app, streaming would significantly improve perceived latency.
3. **Evaluation dataset** — The RAGAS scores are only as good as the judge LLM. A real evaluation set (questions with known correct answers from the transcripts) would give ground-truth recall@k numbers.
4. **PDF support** — Most real earnings transcripts are PDFs from SEC EDGAR, not plain text. Would need `pymupdf` or `pdfplumber` for extraction.
5. **Metadata filtering** — Currently retrieval searches the whole corpus. For a multi-quarter dataset, you'd want to filter by company or date range before semantic search.
