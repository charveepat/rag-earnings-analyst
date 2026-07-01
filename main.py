"""
RAG Earnings Analyst — query any company's earnings call transcript.

Startup flow (no flags needed):
  1. Asks which company / ticker you want to analyze
  2. Auto-fetches the latest earnings transcript from The Motley Fool
  3. Falls back to asking for a URL if auto-fetch fails
  4. Drops into live interactive Q&A

Advanced RAG mode (--mode advanced):
  T1  Hybrid Search       BM25 (keyword) + dense embeddings combined
  T2  HyDE                Generate hypothetical answer doc → embed → retrieve
  T3  Re-ranking          CrossEncoder scores query+doc pairs jointly
  T4  Query Decomposition Break complex questions into atomic sub-queries
  T5  RAG Evaluation      RAGAS-inspired: faithfulness / relevance / precision / recall

Power-user overrides (all optional):
  --source    skip discovery, use this local file or URL
  --ticker    skip the ticker prompt
  --demo      run preset questions instead of interactive mode
  --mode      basic (default) | advanced
  --eval      print RAGAS-style scores after each answer (advanced mode)
  --alpha     hybrid search weight: 0.0=BM25 only, 1.0=dense only (default 0.5)
  --k, --chunk-size, --chunk-overlap, --llm-model, --anthropic-model, --embedding-model

Author: Charvee Patel | github.com/charveepat
"""

import argparse
import json
import logging
import sys
import tempfile
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rank_bm25 import BM25Okapi
    ADVANCED_RAG_AVAILABLE = True
except ImportError:
    ADVANCED_RAG_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "llama3",
    "anthropic_model": "claude-haiku-4-5",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "retriever_k": 4,
    "hybrid_alpha": 0.5,    # 0.0 = pure BM25, 1.0 = pure dense
}

DEMO_QUESTIONS = [
    "What did management say about capex guidance?",
    "Summarize the CFO commentary on margins.",
    "What were the key AI metrics mentioned?",
    "What is the commercial backlog and what does it signal?",
]

# ── Built-in sample dataset (from Session 3) ──────────────────────────────────
# 10 excerpts from Big Tech Q4 FY2024 / Q1 FY2024 / Q4 2023 earnings calls.
# Used by --sample so the tool runs immediately without fetching any transcript.
SAMPLE_EARNINGS = [
    {
        "company": "NVIDIA", "ticker": "NVDA", "quarter": "Q4 FY2024",
        "speaker": "CEO Jensen Huang",
        "text": (
            "Data center revenue reached $18.4 billion, up 409% year over year. "
            "The H100 GPU continues to see extraordinary demand from cloud service "
            "providers and enterprises. We're seeing hyperscalers significantly expand "
            "their AI infrastructure. Gaming revenue was $2.9 billion, up 56% year over "
            "year. We expect Q1 FY2025 revenue to be approximately $24 billion, plus or "
            "minus 2%."
        ),
    },
    {
        "company": "NVIDIA", "ticker": "NVDA", "quarter": "Q4 FY2024",
        "speaker": "CFO Colette Kress",
        "text": (
            "Gross margin reached 76.7% for the quarter, reflecting strong data center "
            "demand and favorable product mix. Operating expenses were $2.5 billion. "
            "We returned $2.5 billion to shareholders through buybacks and dividends. "
            "Our supply chain remains healthy with continued HBM memory improvements. "
            "Free cash flow was $11.2 billion for the quarter."
        ),
    },
    {
        "company": "Microsoft", "ticker": "MSFT", "quarter": "Q2 FY2024",
        "speaker": "CEO Satya Nadella",
        "text": (
            "Azure and other cloud services grew 28% year over year. AI services "
            "contributed 6 percentage points to Azure growth this quarter. Copilot is "
            "being used by 40% of Fortune 500 companies. Commercial cloud revenue "
            "reached $33.7 billion, growing 24% year over year. GitHub Copilot has "
            "over 1.3 million paid subscribers, demonstrating strong developer adoption."
        ),
    },
    {
        "company": "Microsoft", "ticker": "MSFT", "quarter": "Q2 FY2024",
        "speaker": "CFO Amy Hood",
        "text": (
            "Total revenue was $62 billion, up 18% year over year. Operating income "
            "was $27 billion, up 33%. Operating margin expanded to 44%, up 500 basis "
            "points year over year. Capital expenditures were $11.5 billion, driven by "
            "AI and cloud infrastructure. We expect Q3 revenue between $60-61 billion."
        ),
    },
    {
        "company": "Apple", "ticker": "AAPL", "quarter": "Q1 FY2024",
        "speaker": "CEO Tim Cook",
        "text": (
            "Revenue reached $119.6 billion, up 2% year over year. iPhone revenue was "
            "$69.7 billion, roughly flat. Services revenue set an all-time record at "
            "$23.1 billion, up 11% year over year. We now have over 2.2 billion active "
            "devices globally. Installed base reached an all-time high in every "
            "geographic segment, including all-time records in India."
        ),
    },
    {
        "company": "Apple", "ticker": "AAPL", "quarter": "Q1 FY2024",
        "speaker": "CFO Luca Maestri",
        "text": (
            "Gross margin was 45.9%, expanding 50 basis points year over year. Services "
            "gross margin reached 72.8%. We returned over $27 billion to shareholders "
            "during the quarter through buybacks and dividends. Cash position stands at "
            "$162 billion. We expect Q2 revenue to be similar to Q1 levels."
        ),
    },
    {
        "company": "Meta", "ticker": "META", "quarter": "Q4 2023",
        "speaker": "CEO Mark Zuckerberg",
        "text": (
            "2023 was our 'Year of Efficiency.' Operating income grew 156% year over "
            "year. Daily active people across our family of apps reached 3.19 billion, "
            "up 8%. Ad impressions increased 21% year over year, average price per ad "
            "increased 2%. We're increasing 2024 capex guidance to $30-37 billion, "
            "primarily for AI infrastructure investments."
        ),
    },
    {
        "company": "Meta", "ticker": "META", "quarter": "Q4 2023",
        "speaker": "CFO Susan Li",
        "text": (
            "Total revenue was $40.1 billion, up 25% year over year. Operating margin "
            "expanded to 41%, up from 20% in Q4 2022 — a 21 percentage-point improvement. "
            "Total costs and expenses were $23.7 billion, down 8% year over year. "
            "Headcount decreased 22% to 67,317 employees. We expect Q1 2024 revenue "
            "of $34.5-37 billion."
        ),
    },
    {
        "company": "Amazon", "ticker": "AMZN", "quarter": "Q4 2023",
        "speaker": "CEO Andy Jassy",
        "text": (
            "AWS revenue grew 13% year over year to $24.2 billion. We're seeing strong "
            "momentum in AI services through Amazon Bedrock and CodeWhisperer. North "
            "America operating margin reached 6.1%. Advertising services revenue grew "
            "27% year over year to $14.7 billion — our fastest-growing business. We "
            "remain focused on fulfillment cost reduction while accelerating AI investment."
        ),
    },
    {
        "company": "Amazon", "ticker": "AMZN", "quarter": "Q4 2023",
        "speaker": "CFO Brian Olsavsky",
        "text": (
            "Total revenue was $170 billion, up 14% year over year. Operating income "
            "was $13.2 billion, up from $2.7 billion in Q4 2022. Free cash flow over "
            "the trailing 12 months was $32.6 billion. Capital expenditures were "
            "$13.8 billion. We expect Q1 2024 operating income of $8-12 billion."
        ),
    },
]

SAMPLE_DEMO_QUESTIONS = [
    "Which companies showed the strongest revenue growth? Give specific numbers.",
    "What are the AI infrastructure investment trends? Who is spending the most?",
    "Compare operating margins — which company is most profitable?",
    "What did management say about cost-cutting and efficiency initiatives?",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ──────────────────────────────────────────────────────────────
# Sample data loader (Session 3)
# ──────────────────────────────────────────────────────────────

def load_sample_earnings() -> tuple[List[Document], str]:
    """
    Build LangChain Documents from the built-in SAMPLE_EARNINGS dataset.
    Each document carries company/ticker/quarter/speaker metadata so the
    retrieval step can return source-attributed context blocks.
    """
    docs = []
    for entry in SAMPLE_EARNINGS:
        source_label = (
            f"{entry['company']} ({entry['ticker']}) — "
            f"{entry['quarter']}, {entry['speaker']}"
        )
        docs.append(Document(
            page_content=entry["text"],
            metadata={
                "company":  entry["company"],
                "ticker":   entry["ticker"],
                "quarter":  entry["quarter"],
                "speaker":  entry["speaker"],
                "source":   source_label,
            },
        ))

    companies = sorted(set(e["company"] for e in SAMPLE_EARNINGS))
    label = f"Built-in sample ({', '.join(companies)})"
    print(f"\nLoaded {len(docs)} sample excerpts: {label}")
    return docs, label


def format_context_with_sources(docs: List) -> str:
    """
    Build a context block where each chunk is preceded by its source header.
    If no source metadata is present (live transcript), falls back to raw text.

    Example output:
      [NVIDIA (NVDA) — Q4 FY2024, CEO Jensen Huang]
      Data center revenue reached $18.4 billion...
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        header = f"[{source}]\n" if source else ""
        parts.append(f"{header}{doc.page_content}")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────
# Transcript discovery
# ──────────────────────────────────────────────────────────────

def search_transcript_url(company: str) -> Optional[str]:
    """
    Search DuckDuckGo for the most recent Motley Fool earnings transcript
    for the given company name or ticker. Returns a URL or None.
    """
    if not SCRAPING_AVAILABLE:
        return None

    query = f"{company} earnings call transcript site:fool.com"
    search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"

    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            # DDG wraps links — unwrap if needed
            if "uddg=" in href:
                href = urllib.parse.unquote(href.split("uddg=")[-1].split("&")[0])
            if "fool.com" in href and "transcript" in href.lower():
                return href

    except Exception as e:
        logger.warning(f"Transcript search failed: {e}")

    return None


def fetch_fool_transcript(url: str) -> Optional[str]:
    """Fetch and extract the body text from a Motley Fool transcript page."""
    if not SCRAPING_AVAILABLE:
        return None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Motley Fool puts transcript body in .article-body or <article>
        body = (
            soup.find("div", class_="article-body")
            or soup.find("div", attrs={"data-id": "article-body"})
            or soup.find("article")
        )
        if body:
            return body.get_text(separator="\n", strip=True)

        # Fallback: grab all <p> tags
        paragraphs = soup.find_all("p")
        if paragraphs:
            return "\n".join(p.get_text(strip=True) for p in paragraphs)

    except Exception as e:
        logger.warning(f"Failed to fetch transcript from {url}: {e}")

    return None


def resolve_company_name(ticker: str) -> str:
    """Return the long company name for a ticker via yfinance, or the ticker itself."""
    if not YFINANCE_AVAILABLE:
        return ticker
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


def discover_transcript(company_input: str) -> tuple[str, str]:
    """
    Given a company name or ticker, find and fetch the latest earnings transcript.

    Returns (text_content, source_label). Exits on failure if user provides no URL.
    """
    print(f"\nSearching for '{company_input}' earnings call transcript...")

    url = search_transcript_url(company_input)

    if url:
        print(f"Found: {url}")
        print("Fetching transcript...")
        text = fetch_fool_transcript(url)
        if text and len(text) > 500:
            return text, url

    # Auto-fetch failed — ask for a URL
    print("\nCould not auto-fetch the transcript.")
    if not SCRAPING_AVAILABLE:
        print("(Install 'requests' and 'beautifulsoup4' to enable auto-fetch.)")
    url = input("Please paste a URL to the earnings call transcript: ").strip()
    if not url:
        print("No URL provided. Exiting.")
        sys.exit(1)

    print("Fetching transcript from provided URL...")
    try:
        with urllib.request.urlopen(url) as r:
            text = r.read().decode("utf-8", errors="ignore")
        return text, url
    except Exception as e:
        logger.error(f"Failed to fetch transcript from '{url}': {e}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────
# Document loading
# ──────────────────────────────────────────────────────────────

def load_transcript_from_source(source: str) -> List:
    """Load from a local file or URL into LangChain Documents."""
    try:
        if source.startswith("http://") or source.startswith("https://"):
            logger.info(f"Fetching transcript from URL: {source}")
            with urllib.request.urlopen(source) as r:
                text = r.read().decode("utf-8", errors="ignore")
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
            tmp.write(text)
            tmp.close()
            return TextLoader(tmp.name).load()
        else:
            if not Path(source).exists():
                raise FileNotFoundError(f"Transcript not found: {source}")
            return TextLoader(source).load()
    except Exception as e:
        logger.error(f"Failed to load transcript from '{source}': {e}")
        sys.exit(1)


def load_transcript_from_text(text: str) -> List:
    """Write raw text to a temp file and load as LangChain Documents."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(text)
    tmp.close()
    return TextLoader(tmp.name).load()


def split_documents(documents: List, chunk_size: int, overlap: int) -> List:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


# ──────────────────────────────────────────────────────────────
# Basic pipeline: FAISS vector store + Ollama LLM
# ──────────────────────────────────────────────────────────────

def create_vectorstore(chunks: List, model_name: str) -> VectorStore:
    try:
        logger.info("Embedding chunks — this takes ~30 seconds on first run...")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vs = FAISS.from_documents(chunks, embeddings)
        logger.info("Vector store ready")
        return vs
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        sys.exit(1)


def load_llm(model_name: str):
    if not OLLAMA_AVAILABLE:
        logger.error("Ollama unavailable. Use --mode advanced for the Claude-based pipeline.")
        sys.exit(1)
    try:
        llm = Ollama(model=model_name)
        logger.info(f"Loaded LLM: {model_name}")
        return llm
    except Exception as e:
        logger.error(
            f"Failed to load LLM '{model_name}'. Is Ollama running? "
            f"Run: ollama pull {model_name}\nError: {e}"
        )
        sys.exit(1)


def answer_question_basic(question: str, vectorstore: VectorStore, llm, k: int, market_context: str = "") -> str:
    docs = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(question)
    # Use source-attributed context so the LLM knows who said what (Session 3 improvement)
    context = format_context_with_sources(docs)
    extra = f"\n\n{market_context}" if market_context else ""

    prompt = f"""You are a financial analyst assistant. Use the transcript excerpt below \
(and the live market data, if provided) to answer the question. Be concise and specific. \
Cite the company and speaker for every data point you use. \
If the transcript doesn't contain the answer, say so rather than guessing.

Transcript excerpts:
{context}{extra}

Question: {question}
Answer:"""

    return llm.invoke(prompt)


def run_interactive_basic(vectorstore: VectorStore, llm, k: int, market_context: str = "") -> None:
    print("\nAsk your questions below. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        try:
            print(f"A: {answer_question_basic(question, vectorstore, llm, k, market_context)}")
            print("-" * 40)
        except Exception as e:
            logger.error(f"Failed to process question: {e}")


def run_demo_basic(vectorstore: VectorStore, llm, questions: List[str], k: int, market_context: str = "") -> None:
    for question in questions:
        try:
            print(f"\nQ: {question}")
            print(f"A: {answer_question_basic(question, vectorstore, llm, k, market_context)}")
            print("-" * 40)
        except Exception as e:
            logger.error(f"Failed to process question '{question}': {e}")


# ──────────────────────────────────────────────────────────────
# Live market context
# ──────────────────────────────────────────────────────────────

def get_ticker_context(ticker: str) -> str:
    if not YFINANCE_AVAILABLE:
        logger.warning("yfinance not installed — skipping live market context")
        return ""
    try:
        info = yf.Ticker(ticker).info or {}
        if not info:
            return ""
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        pe = info.get("trailingPE")
        market_cap = info.get("marketCap")
        hi = info.get("fiftyTwoWeekHigh")
        lo = info.get("fiftyTwoWeekLow")
        lines = [f"Live market data for {ticker.upper()}:"]
        if price:      lines.append(f"- Current price: ${price}")
        if pe:         lines.append(f"- Trailing P/E: {pe:.2f}")
        if market_cap: lines.append(f"- Market cap: ${market_cap:,}")
        if hi and lo:  lines.append(f"- 52-week range: ${lo} - ${hi}")
        if len(lines) == 1:
            return ""
        logger.info(f"Fetched live market data for {ticker.upper()}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Could not fetch market data for '{ticker}': {e}")
        return ""


# ──────────────────────────────────────────────────────────────
# Advanced pipeline: 5 RAG techniques (from Session 10)
# ──────────────────────────────────────────────────────────────

class AdvancedRAG:
    """
    5-technique RAG pipeline wired to the fetched transcript chunks.

    T1  Hybrid Search       BM25 + dense cosine similarity, normalized + combined
    T2  HyDE                Generate a hypothetical answer passage → embed → retrieve
    T3  Re-ranking          CrossEncoder scores (query, passage) pairs jointly
    T4  Query Decomposition Break complex questions into atomic sub-queries
    T5  RAG Evaluation      RAGAS-inspired: faithfulness / relevance / precision / recall

    Full pipeline per question:
      decompose → hybrid search per sub-query → merge → rerank →
      HyDE verification → final rerank → Claude generates answer → optional eval
    """

    def __init__(self, chunks: List, embedding_model: str, anthropic_model: str):
        if not ADVANCED_RAG_AVAILABLE:
            print("\n⚠  Advanced RAG requires additional packages:")
            print("   pip install sentence-transformers rank-bm25 numpy anthropic")
            sys.exit(1)
        if not ANTHROPIC_AVAILABLE:
            print("\n⚠  Advanced RAG requires the Anthropic SDK:")
            print("   pip install anthropic   (and set ANTHROPIC_API_KEY)")
            sys.exit(1)

        self.chunks = chunks
        self.texts = [c.page_content for c in chunks]
        self.anthropic_model = anthropic_model
        self.client = anthropic.Anthropic()

        print("  Loading sentence-transformers encoder...")
        self.encoder = SentenceTransformer(embedding_model)
        print("  Loading CrossEncoder re-ranker...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("  Building BM25 index...")
        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])
        print("  Building dense embeddings matrix...")
        self.embeddings = self.encoder.encode(self.texts, show_progress_bar=False)
        print(f"  ✓ Advanced RAG ready — {len(self.texts)} chunks indexed\n")

    # ── T1: Hybrid Search ─────────────────────────────────────

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List:
        """
        Combine BM25 (sparse/keyword) and dense cosine similarity.
        alpha=0.0 → pure BM25 | alpha=1.0 → pure dense | alpha=0.5 → balanced
        """
        # Sparse (BM25)
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

        # Dense (cosine similarity)
        q_emb = self.encoder.encode([query])
        cos_sim = np.dot(self.embeddings, q_emb.T).squeeze()
        dense_norm = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-9)

        # Weighted combination
        scores = (1 - alpha) * bm25_norm + alpha * dense_norm
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_idx]

    # ── T2: HyDE (Hypothetical Document Embeddings) ───────────

    def hyde_search(self, query: str, k: int = 5) -> tuple[List, str]:
        """
        Ask Claude to write a hypothetical ideal passage for the query,
        then embed that passage and search — instead of embedding the raw question.
        Works better for abstract or multi-hop questions.
        """
        response = self.client.messages.create(
            model=self.anthropic_model,
            max_tokens=200,
            messages=[{"role": "user", "content": (
                f"Write a 3-sentence excerpt from an earnings call transcript that would "
                f"perfectly answer this analyst question. Be specific with financial language.\n\n"
                f"Question: {query}\n\nWrite only the excerpt, no preamble:"
            )}],
        )
        hypothetical_doc = response.content[0].text.strip()

        h_emb = self.encoder.encode([hypothetical_doc])
        cos_sim = np.dot(self.embeddings, h_emb.T).squeeze()
        top_idx = np.argsort(cos_sim)[::-1][:k]
        return [self.chunks[i] for i in top_idx], hypothetical_doc

    # ── T3: Re-ranking ────────────────────────────────────────

    def rerank(self, query: str, candidates: List, k: int = 4) -> List:
        """
        CrossEncoder reads (query, passage) together — more accurate than
        independent embedding similarity scores. Use after initial retrieval.
        """
        pairs = [[query, c.page_content] for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:k]]

    # ── T4: Query Decomposition ───────────────────────────────

    def decompose_query(self, query: str) -> List[str]:
        """
        Break a complex multi-part question into 2-4 atomic sub-queries.
        Each sub-query retrieves independently, then results are merged before re-ranking.
        """
        response = self.client.messages.create(
            model=self.anthropic_model,
            max_tokens=300,
            messages=[{"role": "user", "content": (
                f"Break this earnings call analyst question into 2-4 specific atomic sub-questions "
                f"that can each be answered independently from transcript text.\n\n"
                f"Question: {query}\n\n"
                f"Return a JSON array of strings only. Example: [\"sub-question 1\", \"sub-question 2\"]"
            )}],
        )
        text = response.content[0].text.strip()
        s, e = text.find("["), text.rfind("]") + 1
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e])
            except json.JSONDecodeError:
                pass
        return [query]  # fallback: treat as single query

    # ── T5: RAG Evaluation ────────────────────────────────────

    def evaluate(self, query: str, answer: str, context_chunks: List) -> dict:
        """
        RAGAS-inspired evaluation. Scores the pipeline on 4 dimensions:
          faithfulness      — is every claim grounded in the retrieved context?
          answer_relevance  — does the answer actually address the question?
          context_precision — are the retrieved chunks relevant to the question?
          context_recall    — does the context contain enough info to answer fully?
        """
        context_text = format_context_with_sources(context_chunks)
        response = self.client.messages.create(
            model=self.anthropic_model,
            max_tokens=500,
            messages=[{"role": "user", "content": f"""Evaluate this RAG response for an earnings call Q&A system.

QUESTION: {query}
RETRIEVED CONTEXT:
{context_text}
GENERATED ANSWER: {answer}

Score each dimension 0.0–1.0:
1. FAITHFULNESS: Is every claim grounded in the context? (1.0 = fully grounded, 0.0 = hallucinated)
2. ANSWER_RELEVANCE: Does the answer address the question? (1.0 = fully answers it)
3. CONTEXT_PRECISION: Are the retrieved chunks relevant? (1.0 = all chunks relevant)
4. CONTEXT_RECALL: Does the context contain enough info? (1.0 = nothing important missing)

Return ONLY valid JSON:
{{"faithfulness": 0.X, "answer_relevance": 0.X, "context_precision": 0.X, "context_recall": 0.X, "reasoning": "one sentence"}}"""}],
        )
        text = response.content[0].text.strip()
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            try:
                scores = json.loads(text[s:e])
                scores["ragas_score"] = np.mean([
                    scores.get("faithfulness", 0),
                    scores.get("answer_relevance", 0),
                    scores.get("context_precision", 0),
                    scores.get("context_recall", 0),
                ])
                return scores
            except (json.JSONDecodeError, KeyError):
                pass
        return {"faithfulness": 0, "answer_relevance": 0, "context_precision": 0,
                "context_recall": 0, "ragas_score": 0, "reasoning": "eval parse error"}

    # ── Full pipeline ─────────────────────────────────────────

    def answer(self, question: str, k: int, alpha: float,
               market_context: str = "", show_eval: bool = False, verbose: bool = False) -> str:
        """
        Full advanced pipeline per question:
          T4 decompose → T1 hybrid search per sub-query → merge (deduplicated) →
          T3 rerank → T2 HyDE verification → T3 final rerank → Claude answer → T5 eval
        """
        # T4: Decompose
        print(f"  [T4] Decomposing query...")
        sub_queries = self.decompose_query(question)
        if verbose:
            for sq in sub_queries:
                print(f"       → {sq}")

        # T1: Hybrid search per sub-query, merge and deduplicate
        print(f"  [T1] Hybrid search ({len(sub_queries)} sub-queries, alpha={alpha})...")
        seen: set = set()
        candidates: List = []
        for sq in sub_queries:
            for chunk in self.hybrid_search(sq, k=k, alpha=alpha):
                if chunk.page_content not in seen:
                    seen.add(chunk.page_content)
                    candidates.append(chunk)

        # T3: Re-rank merged candidates
        print(f"  [T3] Re-ranking {len(candidates)} candidates...")
        top_chunks = self.rerank(question, candidates, k=k)

        # T2: HyDE — generate hypothetical, find any chunks it surfaces that we missed
        print(f"  [T2] HyDE verification...")
        hyde_chunks, hypothetical = self.hyde_search(question, k=3)
        if verbose:
            print(f"       Hypothetical: {hypothetical[:100]}...")
        for c in hyde_chunks:
            if c.page_content not in seen:
                top_chunks.append(c)
        # Final re-rank after merging HyDE results
        top_chunks = self.rerank(question, top_chunks, k=k)

        # Generate answer with Claude — use source-attributed context when available
        context = format_context_with_sources(top_chunks)
        extra = f"\n\n{market_context}" if market_context else ""
        response = self.client.messages.create(
            model=self.anthropic_model,
            max_tokens=600,
            messages=[{"role": "user", "content": (
                f"You are a financial analyst assistant. Use only the transcript excerpts below "
                f"(and live market data if provided) to answer the question. Be concise and specific. "
                f"If the transcript doesn't contain the answer, say so rather than guessing.\n\n"
                f"Transcript excerpts:\n{context}{extra}\n\n"
                f"Question: {question}\nAnswer:"
            )}],
        )
        answer = response.content[0].text.strip()

        # T5: Optional RAGAS-style evaluation
        if show_eval:
            print(f"  [T5] Evaluating answer quality...")
            scores = self.evaluate(question, answer, top_chunks)
            print(f"       Faithfulness:      {scores.get('faithfulness', 0):.2f}")
            print(f"       Answer Relevance:  {scores.get('answer_relevance', 0):.2f}")
            print(f"       Context Precision: {scores.get('context_precision', 0):.2f}")
            print(f"       Context Recall:    {scores.get('context_recall', 0):.2f}")
            print(f"       RAGAS Score:       {scores.get('ragas_score', 0):.2f}")
            print(f"       Reasoning:         {scores.get('reasoning', '')[:120]}")

        return answer

    def run_interactive(self, k: int, alpha: float, market_context: str = "", show_eval: bool = False) -> None:
        print("\nAdvanced RAG active — T1 Hybrid | T2 HyDE | T3 Re-rank | T4 Decompose | T5 Eval")
        print("Ask your questions below. Type 'quit' to exit.\n")
        while True:
            try:
                question = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            try:
                ans = self.answer(question, k=k, alpha=alpha, market_context=market_context,
                                  show_eval=show_eval, verbose=True)
                print(f"\nA: {ans}")
                print("-" * 40)
            except Exception as e:
                logger.error(f"Failed to process question: {e}")

    def run_demo(self, questions: List[str], k: int, alpha: float,
                 market_context: str = "", show_eval: bool = False) -> None:
        for question in questions:
            try:
                print(f"\nQ: {question}")
                ans = self.answer(question, k=k, alpha=alpha, market_context=market_context,
                                  show_eval=show_eval, verbose=True)
                print(f"\nA: {ans}")
                print("-" * 40)
            except Exception as e:
                logger.error(f"Failed to process question '{question}': {e}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG Earnings Analyst — ask questions of any company's earnings call"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Use built-in Big Tech earnings sample (NVDA/MSFT/AAPL/META/AMZN) — no internet needed",
    )
    parser.add_argument(
        "--source", default=None,
        help="Path to a local transcript file, or a URL (skips the company prompt)",
    )
    parser.add_argument(
        "--ticker", default=None,
        help="Stock ticker for live market context (e.g. MSFT). Prompted at startup if omitted.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run the preset demo questions instead of interactive mode",
    )
    parser.add_argument(
        "--mode", choices=["basic", "advanced"], default="basic",
        help="basic: LangChain+FAISS+Ollama | advanced: 5 RAG techniques + Claude (default: basic)",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Print RAGAS-style evaluation scores after each answer (advanced mode only)",
    )
    parser.add_argument(
        "--alpha", type=float, default=CONFIG["hybrid_alpha"],
        help=f"Hybrid search weight: 0.0=BM25 only, 1.0=dense only (default: {CONFIG['hybrid_alpha']})",
    )
    parser.add_argument("--k", type=int, default=CONFIG["retriever_k"],
                        help=f"Transcript chunks to retrieve per question (default: {CONFIG['retriever_k']})")
    parser.add_argument("--chunk-size", type=int, default=CONFIG["chunk_size"],
                        help=f"Chunk size (default: {CONFIG['chunk_size']})")
    parser.add_argument("--chunk-overlap", type=int, default=CONFIG["chunk_overlap"],
                        help=f"Chunk overlap (default: {CONFIG['chunk_overlap']})")
    parser.add_argument("--llm-model", default=CONFIG["llm_model"],
                        help=f"Ollama model name, basic mode only (default: {CONFIG['llm_model']})")
    parser.add_argument("--anthropic-model", default=CONFIG["anthropic_model"],
                        help=f"Claude model, advanced mode only (default: {CONFIG['anthropic_model']})")
    parser.add_argument("--embedding-model", default=CONFIG["embedding_model"],
                        help=f"Embedding model (default: {CONFIG['embedding_model']})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("  RAG EARNINGS ANALYST")
    if args.mode == "advanced":
        print("  Mode: Advanced (T1 Hybrid | T2 HyDE | T3 Re-rank | T4 Decompose | T5 Eval)")
    else:
        print("  Mode: Basic (FAISS + Ollama)")
    print("=" * 60)

    # ── Step 1: resolve source + ticker ──────────────────────
    source_label = args.source
    ticker = args.ticker
    documents = None

    if args.sample:
        # Built-in sample dataset — no internet or transcript needed (Session 3)
        documents, source_label = load_sample_earnings()
        # Sample data covers multiple companies; skip ticker/market context prompt
    elif args.source:
        documents = load_transcript_from_source(args.source)
        if not ticker:
            ticker = input("Ticker for live market data (or press Enter to skip): ").strip() or None
    else:
        company_input = input("\nWhich company's earnings call do you want to analyze?\nEnter ticker or company name: ").strip()
        if not company_input:
            print("No company entered. Exiting.")
            sys.exit(1)

        if company_input.replace(".", "").isalpha() and len(company_input) <= 5:
            ticker = company_input.upper()
        else:
            ticker_guess = company_input.upper().split()[0]
            ticker_prompt = input(f"Ticker for live market data [{ticker_guess}] (or press Enter to skip): ").strip()
            ticker = ticker_prompt.upper() if ticker_prompt else None

        text, source_label = discover_transcript(company_input)
        documents = load_transcript_from_text(text)

    # ── Step 2: chunk ────────────────────────────────────────
    # For sample data the docs are already small excerpts; chunking is still applied
    # so the pipeline is identical for all sources
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)

    # ── Step 3: live market context ──────────────────────────
    market_context = get_ticker_context(ticker) if ticker else ""

    # ── Step 4: header ───────────────────────────────────────
    print("\n" + "=" * 60)
    company_name = resolve_company_name(ticker) if ticker else source_label
    print(f"  {company_name}")
    if ticker:
        print(f"  Ticker: {ticker.upper()}")
    print(f"  Source: {source_label}")
    print(f"  Chunks: {len(chunks)}")
    print("=" * 60)

    # ── Step 5: route to pipeline ─────────────────────────────
    # Use the sample-specific questions when running with --sample --demo
    demo_qs = SAMPLE_DEMO_QUESTIONS if args.sample else DEMO_QUESTIONS

    if args.mode == "advanced":
        rag = AdvancedRAG(chunks, args.embedding_model, args.anthropic_model)
        if args.demo:
            rag.run_demo(demo_qs, k=args.k, alpha=args.alpha,
                         market_context=market_context, show_eval=args.eval)
        else:
            rag.run_interactive(k=args.k, alpha=args.alpha,
                                market_context=market_context, show_eval=args.eval)
    else:
        vectorstore = create_vectorstore(chunks, args.embedding_model)
        llm = load_llm(args.llm_model)
        if args.demo:
            run_demo_basic(vectorstore, llm, demo_qs, args.k, market_context)
        else:
            run_interactive_basic(vectorstore, llm, args.k, market_context)

    logger.info("Session complete")


if __name__ == "__main__":
    main()
