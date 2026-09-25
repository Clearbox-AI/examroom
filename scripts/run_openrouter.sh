#!/usr/bin/env bash
# Run the CRC pilot suite on a selection of OpenRouter models.
#
#   export OPENROUTER_API_KEY=sk-or-...
#   scripts/run_openrouter.sh                      # full runs, recorded to the ledger
#   scripts/run_openrouter.sh --limit 3 --dry-run  # cheap trial, nothing recorded
#
# Any extra arguments are passed through to `examroom run`.
# NOTE: OpenRouter compares model FAMILIES at whatever precision each provider
# serves (usually fp8/bf16). For the quantization ladder of one family (Q4 vs
# Q5 vs FP16), serve GGUFs locally with llama.cpp and use the same openai:
# backend against http://localhost:8080/v1.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first (https://openrouter.ai/keys)}"

SUITE=${SUITE:-suites/crc_pilot_v0.4}
failed=()
MODELS=(
  qwen/qwen3.8-flash
  qwen/qwen3.8-27b
  google/gemma-4-31b-it
  google/gemma-4-26b-a4b-it
  qwen/qwen3-30b-a3b-instruct-2507
  google/gemma-3-27b-it
)

for m in "${MODELS[@]}"; do
  label="or-$(echo "$m" | tr '/:' '--')"
  extra=()
  case "$m" in
    # hybrid-thinking models: without this they spend the whole token budget on
    # hidden reasoning and return empty content (matches anthropic thinking-off default)
    qwen/qwen3.8-*) extra=(--opt reasoning=off) ;;
  esac
  echo "== $m  ($(date -u +%H:%M:%S))"
  # keep going if one model fails (bad id, upstream outage): the others still get recorded
  uv run examroom run "$SUITE" "openai:$m" \
    --opt base_url=https://openrouter.ai/api/v1 \
    --opt api_key_env=OPENROUTER_API_KEY \
    --workers 8 "${extra[@]}" --label "$label" "$@" || { echo "!! FAILED $m"; failed+=("$m"); }
done

uv run examroom report
if ((${#failed[@]})); then echo "failed models: ${failed[*]}"; fi
echo "open report.html for the leaderboard; compare two runs with: examroom diff <id_a> <id_b>"
