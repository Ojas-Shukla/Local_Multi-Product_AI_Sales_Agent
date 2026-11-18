# How Sellbird works

## Local model inference

Every application model request goes through `Brain` to the local Ollama HTTP
server (`OLLAMA_BASE_URL`, default `http://127.0.0.1:11434`). The default model
is `qwen3:1.7b`, configurable with `OLLAMA_MODEL` or `brain.model`. Claude,
Codex, and hosted model APIs are not inference providers in this build; if
Ollama is unavailable or the model is missing, Sellbird pauses and does not
fall back to a hosted model.

The local model handles work that needs judgment, such as writing and
classifying replies. Prospecting, email transport, and other configured
integrations may still make their own network requests.

**What this costs you:**

| | |
|---|---|
| Model inference | **$0 per token** using Ollama locally; your computer supplies the compute |
| Finding businesses | **$0** on the default source, or ~$0.37 per 1,000 on the cheapest paid one |
| Reading their websites | **$0** — three HTTP requests each, no model call |
| Writing and sending | your computer + a mailbox (~$7/mo Google Workspace) |
| Prospecting tools | **none.** No Apollo, no ZoomInfo, no Clearbit, no Clay. |

There is no hosted model subscription or usage quota to read. Sellbird tracks
local model calls in its usage view; hardware capacity and Ollama's availability
determine inference speed.

---

## What Sellbird actually does

Sellbird runs a heartbeat: wake up, check the budget, decide what most needs doing, do it, log it, sleep. Every 15 minutes by default.

```
                 ┌──────────────────────────────────────┐
                 ↓                                      │
   Wake  →  Quiet hours?  →  Budget OK?  →  Decide  →  Act  →  Log  →  Sleep
                 │               │
              (sleep)         (sleep)
```

It decides deterministically, not by asking a model what to do — that would burn a call on something derivable from four counts:

```
handle replies  >  send campaigns  >  write campaigns  >  prospect  >  idle (analyze)
```

Profiling and outbox draining ride along on **every** cycle regardless, because they cost nothing and have to happen on schedule.

## Project structure

```
sellbird/
├── main.py              # heartbeat loop
├── brain.py             # Ollama-only inference routing, skills + usage
├── local_ai/            # Ollama HTTP client and hardware recommendations
├── state.py             # SQLite schema, migrations, observations, cohorts
├── signals.py           # the 23-signal catalog Sellbird proposes
├── pipeline.py          # DISCOVER → PROFILE chaining
├── gate.py              # deterministic pre-send checks
├── dashboard.py         # FastAPI JSON API
├── dashboard_auth.py    # dashboard token, Host/Origin checks
├── collectors/
│   ├── discover.py      # provider adapters, junk filtering, entity resolution
│   └── profile.py       # the free HTTP profiler
├── agents/              # scout, writer, sender, handler, analyst
├── integrations/        # mail providers, email finder, quota
└── web/                 # dashboard HTML/CSS/JS — no build step

skills/                  # ← the sales knowledge. Editable Markdown.
prompts/                 # ← agent system prompts. Also editable Markdown.
tests/                   # 242 tests
```

---

## Philosophy

**Autonomous doesn't mean unsupervised.** Quiet hours, spend caps, send limits, an approval queue, a deterministic pre-send gate, and a kill switch. Sellbird asks before it spends and before it sends, until you tell it not to.

**You decide what a good prospect is.** Sellbird proposes signals; you confirm them. A prospect list you can't explain is a prospect list you shouldn't send to.

**If it can be gotten deterministically, don't use a model.** Rank position, tech detection, junk filtering, the priority decision, and the pre-send gate are plain code. The local Ollama model is reserved for the parts that need judgment.

**A failure is an observation, not silence.** "No team page found" is a finding. "Overpass is throttling" is a finding. Silent failures look exactly like clean results, which is how a broken run gets mistaken for an empty market.

**The database is the asset, not the agent.** The initial list is worth little. The same signals observed over quarters — who changed agencies, who started spending on ads, who let their content go stale — is worth a great deal.

**Everything is editable.** Prompts, skills, config, the dashboard. All plain text. You don't need to be a developer to change how Sellbird sells.
