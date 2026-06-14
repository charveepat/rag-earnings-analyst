"""
RAG Earnings Analyst — query any company's earnings call transcript.

Startup flow (no flags needed):
  1. Asks which company / ticker you want to analyze
  2. Auto-fetches the latest earnings transcript from The Motley Fool
  3. Falls back to asking for a URL if auto-fetch fails
  4. Drops into live interactive Q&A

Power-user overrides (all optional):
  --source   skip discovery, use this local file or URL
  --ticker   skip the ticker prompt
  --demo     run preset questions instead of interactive mode
  --k, --chunk-size, --chunk-overlap, --llm-model, --embedding-model

Author: Charvee Patel | github.com/charveepat
"""

import argparse
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_community.llms import Ollama

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


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "llama3",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "retriever_k": 4,
}

DEMO_QUESTIONS = [
    "What did management say about capex guidance?",
    "Summarize the CFO commentary on margins.",
    "What were the key AI metrics mentioned?",
    "What is the commercial backlog and what does it signal?",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


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
# Vector store & LLM
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


def load_llm(model_name: str) -> Ollama:
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
# Q&A pipeline
# ──────────────────────────────────────────────────────────────

def answer_question(question: str, vectorstore: VectorStore, llm: Ollama, k: int, market_context: str = "") -> str:
    docs = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    extra = f"\n\n{market_context}" if market_context else ""

    prompt = f"""You are a financial analyst assistant. Use the transcript excerpt below \
(and the live market data, if provided) to answer the question. Be concise and specific. \
If the transcript doesn't contain the answer, say so rather than guessing.

Transcript excerpt:
{context}{extra}

Question: {question}
Answer:"""

    return llm.invoke(prompt)


def run_interactive(vectorstore: VectorStore, llm: Ollama, k: int, market_context: str = "") -> None:
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
            print(f"A: {answer_question(question, vectorstore, llm, k, market_context)}")
            print("-" * 40)
        except Exception as e:
            logger.error(f"Failed to process question: {e}")


def run_demo_questions(vectorstore: VectorStore, llm: Ollama, questions: List[str], k: int, market_context: str = "") -> None:
    for question in questions:
        try:
            print(f"\nQ: {question}")
            print(f"A: {answer_question(question, vectorstore, llm, k, market_context)}")
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
        "--source",
        default=None,
        help="Path to a local transcript file, or a URL (skips the company prompt)",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Stock ticker for live market context (e.g. MSFT). Prompted at startup if omitted.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the preset demo questions instead of interactive mode",
    )
    parser.add_argument("--k", type=int, default=CONFIG["retriever_k"],
                        help=f"Transcript chunks to retrieve per question (default: {CONFIG['retriever_k']})")
    parser.add_argument("--chunk-size", type=int, default=CONFIG["chunk_size"],
                        help=f"Chunk size (default: {CONFIG['chunk_size']})")
    parser.add_argument("--chunk-overlap", type=int, default=CONFIG["chunk_overlap"],
                        help=f"Chunk overlap (default: {CONFIG['chunk_overlap']})")
    parser.add_argument("--llm-model", default=CONFIG["llm_model"],
                        help=f"Ollama model name (default: {CONFIG['llm_model']})")
    parser.add_argument("--embedding-model", default=CONFIG["embedding_model"],
                        help=f"Embedding model (default: {CONFIG['embedding_model']})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("  RAG EARNINGS ANALYST")
    print("=" * 60)

    # ── Step 1: resolve source + ticker ──────────────────────
    source_label = args.source
    ticker = args.ticker
    documents = None

    if args.source:
        # Power-user path: source provided via flag
        documents = load_transcript_from_source(args.source)
        if not ticker:
            ticker = input("Ticker for live market data (or press Enter to skip): ").strip() or None
    else:
        # Discovery path: ask the user
        company_input = input("\nWhich company's earnings call do you want to analyze?\nEnter ticker or company name: ").strip()
        if not company_input:
            print("No company entered. Exiting.")
            sys.exit(1)

        # Use input as ticker if it looks like one, otherwise ask separately
        if company_input.replace(".", "").isalpha() and len(company_input) <= 5:
            ticker = company_input.upper()
        else:
            # Try to guess ticker from yfinance, but don't block on failure
            ticker_guess = company_input.upper().split()[0]
            ticker_prompt = input(f"Ticker for live market data [{ticker_guess}] (or press Enter to skip): ").strip()
            ticker = ticker_prompt.upper() if ticker_prompt else None

        text, source_label = discover_transcript(company_input)
        documents = load_transcript_from_text(text)

    # ── Step 2: chunk + embed ────────────────────────────────
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
    vectorstore = create_vectorstore(chunks, args.embedding_model)

    # ── Step 3: LLM ─────────────────────────────────────────
    llm = load_llm(args.llm_model)

    # ── Step 4: live market context ──────────────────────────
    market_context = get_ticker_context(ticker) if ticker else ""

    # ── Step 5: header ───────────────────────────────────────
    print("\n" + "=" * 60)
    company_name = resolve_company_name(ticker) if ticker else source_label
    print(f"  {company_name}")
    if ticker:
        print(f"  Ticker: {ticker.upper()}")
    print(f"  Source: {source_label}")
    print("=" * 60)

    # ── Step 6: Q&A ─────────────────────────────────────────
    if args.demo:
        run_demo_questions(vectorstore, llm, DEMO_QUESTIONS, args.k, market_context)
    else:
        run_interactive(vectorstore, llm, args.k, market_context)

    logger.info("Session complete")


if __name__ == "__main__":
    main()
