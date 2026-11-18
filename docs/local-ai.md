# Local AI inference

This assignment build uses Ollama as the only application LLM provider.

## Request path

```text
Sales agents / workflows
        |
        v
sellbird.brain.Brain
        |
        v
sellbird.local_ai.OllamaClient
        |
        | HTTP POST /api/chat
        v
http://127.0.0.1:11434
        |
        v
qwen3:1.7b (local Ollama model)
```

`BrainConfig` accepts only `provider: ollama`. Claude, Codex, OpenAI, Gemini,
and other hosted LLM providers are rejected by configuration validation.
There is no hosted LLM fallback path in `Brain.think`.

## Configure Ollama

```bash
ollama pull qwen3:1.7b
ollama list
cp .env.example .env
```

Set the following values in `.env` when the defaults are not appropriate:

```dotenv
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_TIMEOUT_SECONDS=120
```

The `.env` file is loaded by the Ollama settings reader, including when the
application is started from a CLI path that does not load the full application
configuration first.

## Failure behaviour

- If the Ollama server cannot be reached, the local AI client raises an
  `OllamaUnavailable` error.
- If the configured model is not installed, it raises an
  `OllamaModelNotFound` error and prints the matching `ollama pull` command.
- The Brain records the local provider as paused and returns an empty result
  to its caller. It does not call a hosted model.
- The deterministic, non-LLM parts of the existing application remain separate
  from this inference client.

## Execution evidence

Successful calls write `LOCAL INFERENCE` log entries with the provider, model,
agent, task, and elapsed time. Local call counts are stored in SQLite under
`usage_log.ollama_calls`; per-call attribution is stored in `usage_events` with
`provider=ollama` and `source=ollama_local`.

## Tests

```bash
python -m pytest -q tests/test_ollama_client.py tests/test_brain_ollama.py
```

The Ollama client tests use an HTTP mock and do not require a running Ollama
server. The Brain tests verify local routing, provider validation, failure
status, and the SQLite usage-counter migration. A real model check is performed
by the setup wizard and can also be run with `sellbird status --check`.

## Hardware awareness and model sizing

The application detects operating system/architecture, logical CPU count,
physical CPU cores when the OS exposes them, installed RAM, and GPU name/VRAM
when supported local tools are available (`nvidia-smi`, `rocm-smi`, macOS
`system_profiler`, or Windows PowerShell/CIM). Detection is local and does not
make network requests. If no supported tool reports a GPU, the response says
that a GPU was not detected; this is not proof that no other accelerator is
present.

The hardware endpoint is `GET /api/system/hardware` (served by the local
dashboard). It reports both the recommended tier and the configured model.
The standard profile recommends `qwen3:1.7b` for ordinary laptops and CPU-only
systems, with `num_ctx=4096`, `num_predict=512`, `temperature=0.2`, and a
`5m` keep-alive. The high-performance profile is selected conservatively at
32+ GB system RAM plus 8+ GB detected NVIDIA VRAM and recommends `qwen3:8b`
with `num_ctx=8192`, `num_predict=1024`, `temperature=0.2`, and a `10m`
keep-alive as an informational larger-model tier. This recommendation never
changes `OLLAMA_MODEL`: the assignment's configured model remains
`qwen3:1.7b` unless the operator explicitly changes it.

## Ollama outage behaviour and deferred tasks

If local Ollama is unavailable or the configured model is missing, the Brain
marks local inference as paused, logs the clear reason, and does not contact a
hosted model. Non-LLM application functions remain available where their
existing code permits. An inference request that fails while Ollama is
unavailable is stored in SQLite (`local_ai_task_queue`) with its prompt, agent,
task, JSON expectation, status, attempts, result, and last error.

After Ollama is restored, call `Brain.process_queued_tasks()` to retry pending
requests. Successful responses are stored on their queue rows; failed retries
remain pending. Queue processing uses only the configured local Ollama model.


## Business profile and product-specific ICP

The dashboard's **Your company** section now supports two additional local workflows:

- **Business profile generation** reads the configured company identity, the local product catalogue, and locally stored product/competitor knowledge. It asks the configured Ollama model for a structured profile and stores the result in the local SQLite `settings` table. Unknown facts are kept in an explicit `unknowns` list rather than being invented.
- **Product-specific ICP generation** runs separately for each saved product. Each product stores its own ICP fields (industries, company size, buyer titles, geography, pain points, buying triggers, disqualifiers, and rationale), so generating an ICP for one product does not overwrite another product's ICP.

The first catalogue entry is seeded from the existing `sellbird.yaml` product and ICP configuration. Existing **What you sell** and **Who you sell to** edits continue to update that first entry. Additional products are stored locally and can have their own ICPs. Both generation endpoints use `Brain.think_json` and the configured local Ollama model; there is no cloud fallback. Generated records include the model/provider and timestamp for traceability.


## Prospect research from previously downloaded data

The Contacts table provides a **Research** action for an existing prospect. The workflow reads
the saved contact and company records and previously collected SQLite observations. It does not
fetch the company website, run a new search, or call a prospecting/enrichment service. This makes
the local research workflow usable after the initial prospect data has been downloaded and while
the computer is offline.

The selected local Ollama model produces a structured research summary, relevant facts,
potential relevance hypotheses, research gaps, suggested questions, a product-fit note, and a
confidence label. Evidence references shown in the UI are assembled from source URLs already
present in the local records rather than accepting model-invented URLs. Results are persisted in
SQLite in `prospect_research` and tagged with `provider=ollama_local`, model, source mode, and
generation timestamp. If Ollama is unavailable, the API returns an explicit error and Brain's
existing local-AI queue records the failed inference request; no cloud model is contacted.

API endpoints: `GET /api/prospects/{prospect_id}/research` retrieves the saved result;
`POST /api/prospects/{prospect_id}/research` generates or refreshes it. An optional `product_id`
can attach the matching saved product ICP to the research prompt.

## Buying-signal classification

The Contacts table exposes a **Signals** action for each saved prospect. The classifier
uses only the prospect/company fields and observations already stored in SQLite, plus
previously generated local prospect research and the selected product's saved ICP when
provided. It does not crawl websites, run fresh discovery, or call a prospect-enrichment
service. Qwen3 1.7B is invoked through the existing local Ollama Brain path.

The result records overall signal strength, individual signal types and strength,
buying stage, urgency, product relevance, recommended next action, confidence, and
missing evidence. Evidence dates and source links are accepted only when they match a
locally stored observation; model-invented dates and URLs are not displayed as verified
sources. Classifications are persisted by prospect and product in SQLite and can be
retrieved through `GET /api/prospects/{prospect_id}/buying-signals?product_id=...`.

Generate or refresh a classification with `POST /api/prospects/{prospect_id}/buying-signals`
using an optional JSON body `{"product_id":"..."}`. When Ollama is unavailable, the
existing local-AI queue/error handling applies; there is no cloud inference fallback.


## Personalised email generation

In Contacts, choose **Email** for a prospect. Optionally select a product to provide product description, benefits and product-specific ICP context. Generation uses only locally stored prospect/company fields, saved local research and saved buying-signal classification for that product. The result contains a subject, body, personalisation basis and evidence gaps. The result is persisted in SQLite and marked `draft_only`.

The workflow does not send email, create an external mailbox draft, fetch prospect websites, or call a paid service. Review every generated draft before using it.

API:

- `GET /api/prospects/{prospect_id}/personalised-email?product_id=...`
- `POST /api/prospects/{prospect_id}/personalised-email` with `{"product_id":"..."}`

## Reply classification

In Contacts, choose **Reply**, paste a sample conversation or the prospect's latest reply, and optionally select a product. Local Ollama classifies intent (`interested`, `objection`, `not_interested`, `out_of_office`, `wrong_person`, `unsubscribe`, `referral`, `needs_follow_up`, or `neutral`), sentiment, buying readiness, key points, and a recommended next action. It may provide a suggested response draft. Results are saved locally.

The conversation is treated as untrusted input, not as instructions to the model. No response is sent automatically.

API:

- `GET /api/prospects/{prospect_id}/reply-classification?product_id=...`
- `POST /api/prospects/{prospect_id}/reply-classification` with `{"product_id":"...","conversation_text":"..."}`

## Local inference and persistence

Both workflows use the existing `Brain.think_json` execution path with the `ollama` provider and configured local model. SQLite migration v21 creates `personalised_email_drafts` and `reply_classifications`, keyed by prospect and product. Provider metadata is `ollama_local`.

No email is sent by these endpoints. No external AI endpoint, paid inference endpoint, or web-enrichment API is called. Internet access is not required after the application and model are installed.

## Verification

- `python -m compileall -q sellbird tests`
- `node --check sellbird/web/app.js`
- SQLite migration scripts 1–21 executed successfully against an in-memory SQLite database.
- Phase 7 service tests cover local model task routing, product context, persistence, input validation and invalid model output.

Run in the project's normal virtual environment:

```bash
python -m pytest -q tests/test_outreach.py
```
