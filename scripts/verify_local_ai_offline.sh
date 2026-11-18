#!/usr/bin/env bash
# Local-only preflight: checks configuration, Ollama loopback health and model presence.
# This script does not toggle networking and does not prove the offline acceptance test.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail() { printf 'FAIL: %s\n' "$1" >&2; exit 1; }
pass() { printf 'PASS: %s\n' "$1"; }

[[ -f .env ]] || fail ".env is missing; copy .env.example to .env and set local Ollama values"

OLLAMA_BASE_URL="$(grep -E '^OLLAMA_BASE_URL=' .env | tail -n1 | cut -d= -f2- | tr -d '\r"' || true)"
OLLAMA_MODEL="$(grep -E '^OLLAMA_MODEL=' .env | tail -n1 | cut -d= -f2- | tr -d '\r"' || true)"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:1.7b}"

case "$OLLAMA_BASE_URL" in
  http://127.0.0.1:*|http://localhost:*|http://[::1]:*) pass "Ollama endpoint is loopback: $OLLAMA_BASE_URL" ;;
  *) fail "Ollama endpoint is not loopback ($OLLAMA_BASE_URL). Set OLLAMA_BASE_URL=http://127.0.0.1:11434 for this local-only assignment." ;;
esac

command -v ollama >/dev/null 2>&1 || fail "ollama command not found; install Ollama first"
command -v curl >/dev/null 2>&1 || fail "curl is required for the localhost health check"

TAGS_URL="${OLLAMA_BASE_URL%/}/api/tags"
TAGS="$(curl --fail --silent --show-error --max-time 5 "$TAGS_URL")" || fail "Cannot reach local Ollama at $TAGS_URL; start ollama serve"
python3 -c 'import json,sys; d=json.load(sys.stdin); print("\\n".join(x.get("name", "") for x in d.get("models", [])))' <<< "$TAGS" | grep -Fxq "$OLLAMA_MODEL" || fail "Configured model '$OLLAMA_MODEL' is not listed. Run: ollama pull $OLLAMA_MODEL"
pass "Configured model is installed: $OLLAMA_MODEL"

if grep -Eq '^(OPENAI_API_KEY|ANTHROPIC_API_KEY|GEMINI_API_KEY|CLAUDE_API_KEY|CODEX_API_KEY)=[^[:space:]]+' .env; then
  printf 'WARN: one or more cloud LLM credential variables are populated in .env. They are not used by this local Brain path, but remove unused credentials for a cleaner audit.\n'
else
  pass "No common cloud LLM API key variables are populated in .env"
fi

printf '\nLocal AI preflight passed. This is not the manual offline acceptance result.\n'
printf 'Next: disconnect internet and follow docs/OFFLINE_ACCEPTANCE.md.\n'
