# Final offline acceptance and submission readiness

## Acceptance objective

Prove that, after setup and the model download, all eight required local AI workflows execute using the local Ollama model while internet access is disabled. This is a **manual real-model acceptance test**. Passing pytest alone does not prove offline inference.

## Before disconnecting the internet

1. Install the project dependencies and complete first-run setup.
2. Install Ollama and download the exact configured model:

   ```bash
   ollama pull qwen3:1.7b
   ollama list
   ```

3. Copy `.env.example` to `.env` if needed. Confirm these local settings:

   ```dotenv
   OLLAMA_BASE_URL=http://127.0.0.1:11434
   OLLAMA_MODEL=qwen3:1.7b
   OLLAMA_TIMEOUT_SECONDS=120
   ```

4. Prepare the sample company, two products, product-specific ICPs, one or more prospects, and locally stored prospect/company observations. Do not rely on live discovery during the offline run.
5. Start Ollama and warm the model once while online:

   ```bash
   ollama serve
   # In another terminal:
   ollama run qwen3:1.7b "Return the word READY."
   ```

6. Run the preflight script while the local Ollama service is reachable:

   ```bash
   bash scripts/verify_local_ai_offline.sh
   ```

7. Capture the preflight output, `ollama list`, dependency versions, and a copy of the sample dataset (with personal data removed).

## Offline acceptance procedure

1. Disconnect the machine from Wi-Fi/Ethernet or disable its network adapter. Keep loopback networking (`127.0.0.1`) enabled. Do not use a firewall rule that blocks localhost.
2. Keep the Ollama server running locally. Start Sellbird in a separate terminal:

   ```bash
   sellbird dashboard
   ```

   The CLI prints the local dashboard URL; the documented default is `http://127.0.0.1:5555`.
3. In the dashboard, use the already saved local company/product/prospect data. Complete and record each required workflow:

   | # | Required workflow | Offline source/input | Evidence |
   |---|---|---|---|
   | 1 | Business profile generation | Saved company details and local product catalogue | Generated profile and local inference log |
   | 2 | Product-specific ICP generation | Saved product details | ICP for each product, showing separation |
   | 3 | Prospect research | Previously downloaded prospect/company records and saved observations | Research result and `previously_downloaded_local_data` source mode |
   | 4 | Buying-signal classification | Local prospect facts, saved research and selected product ICP | Classification with evidence gaps |
   | 5 | Problem-statement generation | Local prospect context and selected product | Problem statement and discovery questions |
   | 6 | Personalised email generation | Local prospect context, research and buying signals | Draft marked `draft_only`; no message sent |
   | 7 | Prospect scoring | Local facts and selected product ICP | AI score shown separately from Scout score |
   | 8 | Reply classification | Manually supplied sample conversation | Intent and recommended next action; no reply sent |

4. Refresh the dashboard and confirm the saved outputs remain available. Switch to another product and verify that its outputs are independent.
5. Inspect application logs for entries similar to:

   ```text
   LOCAL INFERENCE completed provider=ollama model=qwen3:1.7b ...
   ```

   Capture the real log lines. Do not fabricate or edit logs.
6. Confirm the app did not require internet access for any of the eight workflows. If any step fails, record the exact failure; do not label the offline acceptance as passed.
7. Reconnect internet only after the test is complete.

## Automated regression checks

Run with the project's virtual environment activated:

```bash
python -m pytest -q \
  tests/test_state.py \
  tests/test_prospect_research.py \
  tests/test_prospect_intelligence.py \
  tests/test_buying_signals.py \
  tests/test_outreach.py \
  tests/test_local_sales_workflow.py \
  tests/test_dashboard_ai_workflows.py

python -m pytest -q
node --check sellbird/web/app.js
bash scripts/verify_local_ai_offline.sh
```

The deterministic workflow integration tests use a fake Brain and real temporary SQLite storage. They validate persistence, product isolation and workflow wiring, but do not replace the manual test with a real Qwen3 model and the network disconnected.

## Evidence checklist

- [ ] `ollama list` showing `qwen3:1.7b` installed
- [ ] Preflight script output
- [ ] Full pytest output and JavaScript syntax-check output
- [ ] Company profile and two product-specific ICPs
- [ ] Prospect research using saved local data
- [ ] Buying signals and problem statement
- [ ] AI score beside the unchanged Scout score
- [ ] Email draft marked not sent
- [ ] Sample reply classification
- [ ] Local inference logs naming provider and model
- [ ] Recording showing all eight workflows with internet disconnected
- [ ] Known limitations and any failed/partial steps disclosed

## Important boundaries

- First-time package installation and model download require connectivity. The acceptance test starts only after those are complete.
- Fresh prospect discovery, public-web crawling, website research, email-provider access, and optional third-party enrichment are separate internet-dependent capabilities. Do not invoke them during the offline acceptance test.
- Local inference is routed through the local Ollama client. The application may include optional integrations for non-LLM tasks; their presence does not make those integrations part of the offline AI acceptance path.
- The project does not claim that every optional integration or every application feature works offline.
- A local dashboard can use `127.0.0.1` while the machine has no internet connection.
