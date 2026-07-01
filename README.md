# Course Materials RAG System — fixed fork

This is the starter codebase from the DeepLearning.AI "Claude Code" course. I was
working through the course, went to run it, and every query came back as a 500.
Turns out a few things were built against an older Claude model: a retired model ID,
a tool-use loop that only survived one search, and a reply-token budget too small
for how newer models "think." I tracked them down, fixed them, and put the working
version up here so anyone else stuck on the same course can skip the headache.

It runs on current Claude models up to Claude Sonnet 5 today. A future model might
need another small tweak — the model ID lives in one spot (`backend/config.py`), and
the fixes below explain the gotchas — but nothing here is hard-wired to one specific
model.

If your queries are failing with `Error: Query failed` or a 500, jump to
[Fixes applied](#fixes-applied) — it explains exactly what was wrong and what I
changed.

Before you run it: you need your own Anthropic API key (goes in `.env`, see below).
That's it — the four DeepLearning.AI course transcripts are already in `docs/`, so
the app answers real questions the moment you add your key. Those transcripts are
DeepLearning.AI's, pulled from their public starter repo — see
[Course materials](#course-materials) for the source and credit.

## What it does

A small full-stack RAG app: ask questions about course materials and get back
answers with their sources. ChromaDB handles the vector search, Claude writes the
answers, and there's a plain web UI on a FastAPI backend.

## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key (for Claude AI)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**

   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Course materials

The `docs/` folder ships with the four course transcripts from the DeepLearning.AI
starter codebase — Building Towards Computer Use, MCP: Build Rich-Context AI Apps,
Advanced Retrieval with Chroma, and Prompt Compression & Query Optimization — so the
app has real content to answer from the moment you clone it. Credit for that content
goes to DeepLearning.AI; it comes from their public starter repo:
https://github.com/https-deeplearning-ai/starting-ragchatbot-codebase

Want to add your own course? Drop any plain-text `.txt` file into `docs/` and it's
ingested into ChromaDB on startup. One file per course, starting with metadata lines
like:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: Introduction
Lesson Link: <url>
<transcript text...>
```

## Fixes applied

Out of the box, every query to `POST /api/query` fails with current Claude models.
There were three separate problems behind it:

1. **The model ID was retired.** `backend/config.py` pointed at
   `claude-sonnet-4-20250514`, which Anthropic no longer serves, so the API just
   returns `404 not_found_error`. I pointed it at a current model
   (`claude-sonnet-5`) — set this to whatever your own API key can access. Heads up:
   newer models (Sonnet 5, Opus 4.7+) also reject the `temperature` parameter with a
   `400`, so I removed the hardcoded `temperature: 0` in `backend/ai_generator.py`.

2. **The tool-use loop only handled one search, then crashed.**
   `backend/ai_generator.py` ran the search tool exactly once, then re-called the
   model *without* tools and grabbed `response.content[0].text`. Newer models often
   want to search a second time; with no tools available they hand back an empty
   turn, so `content[0]` blows up with `IndexError` and the endpoint 500s. I
   reworked `_handle_tool_execution` into a bounded loop that:
   - keeps the tools available across rounds so the model can search as many times
     as it actually needs,
   - drops the tools on the final round so the model has to give a text answer, and
     never leaves a dangling `tool_use` behind (the API rejects that with a `400`),
   - on that final round, explicitly tells the model to answer from what it already
     found. Without this it would sometimes still "want" another search, get no
     tools, and hand back an empty turn — which showed up as a flaky "couldn't
     generate a response" on roughly 1 in 3 broad questions.
   - falls back to a plain message if a response somehow still comes back empty.

3. **The reply-token budget was too small for newer models.**
   `backend/ai_generator.py` capped every reply at `max_tokens: 800`. Claude
   Sonnet 5 runs *adaptive thinking* by default, and those thinking tokens count
   against that same cap. On a longer answer — a course outline, say — the model
   could spend the whole 800 tokens thinking and hand back a turn with no text in it
   at all, which showed up as an intermittent "I wasn't able to generate a
   response." I raised the cap to `4096` so the thinking *and* a full answer fit, and
   made the text extraction join every text block in the turn instead of grabbing
   only the first one.

If you still hit a 404 on queries, your key probably can't access that model. Run
this to list the models you *can* use, then update `ANTHROPIC_MODEL` in
`backend/config.py`:

```bash
cd backend && uv run python -c "from config import config; import anthropic; print([m.id for m in anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY).models.list().data])"
```
