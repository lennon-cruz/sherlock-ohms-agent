# Sherlock Ohms Agent

A small agentic RAG assistant. It has two tools, each backed by its own vector store built
from an industry report:

- `search_fintech_report` — World Bank, *Fintech and the Future of Finance* (2023)
- `search_telecom_report` — ITU, *Measuring Digital Development: Facts and Figures* (2024)

Given a question, the agent decides which report (if any) is relevant, retrieves the most
similar chunks from that report, and answers grounded in that context.

## Architecture

```
query -> Agent (OpenAI tool-calling loop)
           |
           +-- search_fintech_report(query) -> RAG(fintech collection) -> answer
           +-- search_telecom_report(query) -> RAG(telecom collection) -> answer
```

- `src/sherlock_ohms/tools.py` — turns a plain Python function into an OpenAI tool schema
- `src/sherlock_ohms/agent.py` — the tool-calling loop (no external agent framework)
- `src/sherlock_ohms/rag.py` — retrieve top-k chunks, then answer from that context only
- `src/sherlock_ohms/vectorstore.py` — Chroma collection backed by OpenAI embeddings
- `src/sherlock_ohms/ingest.py` — downloads the two reports and embeds them

## Setup

Requires [uv](https://docs.astral.sh/uv/).

### 1. Get an OpenAI API key

1. Create a free account at [platform.openai.com](https://platform.openai.com/signup) (or sign in if you already have one).
2. Go to the [API keys page](https://platform.openai.com/api-keys) and click **Create new secret key**.
3. Copy the key — OpenAI only shows it once.

> **Never commit or push your API key.** Keep it only in your local `.env` file, which is
> already gitignored. If you ever paste a key into a file tracked by git, rotate/revoke it
> from the API keys page and generate a new one.

### 2. Install and run

```bash
uv sync
cp .env.example .env   # then paste your OPENAI_API_KEY into .env
uv run ingest          # downloads both PDFs and builds the local vector stores
uv run agent           # interactive REPL
```

## Corpus licensing

- World Bank fintech report: CC BY 3.0 IGO
- ITU telecom report: CC BY-NC-SA 3.0 IGO

Both PDFs are downloaded on demand by `uv run ingest` (not committed to this repo) and are
used here for non-commercial, informational purposes with attribution.
