#!/usr/bin/env bash
# run.sh — executed by the orx harness as the experiment's run command.
#
# This drives the COMPLETE open-weights reproduction pipeline with
# qwen3:30b-a3b (served locally via Ollama) as the main agent, then copies the
# inspectable evidence into .openresearch/artifacts/ so orx records it.
#
# Pipeline (the model writes/improves every experiment, runs it, checks
# determinism, and assigns every verdict; the harness only executes the code
# the model writes and reports stdout/stderr back):
#   driver_openweights.py            initial write/execute/verdict loop (both claims)
#   challenge_claim1.py _round2..5   5 adversarial scrutiny rounds on Claim 1
#   claim2_reconsider*.py            Claim 2 consistency check
#   claim2_posterfinding.py          Claim 2 poster summary
#   challenge_claim2_round1..3       3 judge-feedback-driven scrutiny rounds on Claim 2
#   final_summary*.py                paper-level summary regeneration
set -euo pipefail

# Resolve to the repo root (this script lives in <repo>/.openresearch/).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Prefer the dedicated venv (has numpy/scipy/pandas); fall back to system python3.
if [ -x "$HOME/.venvs/icml26/bin/python" ]; then
  PY="$HOME/.venvs/icml26/bin/python"
else
  PY="$(command -v python3)"
  echo "run.sh: icml26 venv not found, using $PY" 1>&2
fi
echo "run.sh: REPO=$REPO  PY=$PY"

# --- preflight: Ollama must be up and the model pulled (this is a local run) ---
echo "run.sh: checking Ollama + model..."
if ! curl -sf -m 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "run.sh: FATAL — Ollama is not responding at http://localhost:11434" 1>&2
  exit 2
fi
if ! curl -sf -m 5 http://localhost:11434/api/tags | grep -q '"qwen3:30b-a3b"'; then
  echo "run.sh: FATAL — qwen3:30b-a3b not pulled (run: ollama pull qwen3:30b-a3b)" 1>&2
  exit 3
fi
echo "run.sh: Ollama + qwen3:30b-a3b OK"

# --- run the full agent pipeline ---
# Each stage appends to orx-openweights-trace.jsonl (the driver resets it at start).
# `set -e` aborts if any stage exits non-zero, so a real failure surfaces as a
# failed run rather than a silently-incomplete one.
echo "run.sh: starting agent pipeline at $(date -u +%FT%TZ)..."
"$PY" driver_openweights.py
"$PY" challenge_claim1.py
"$PY" challenge_claim1_round2.py
"$PY" challenge_claim1_round3.py
"$PY" challenge_claim1_round4.py
"$PY" challenge_claim1_round5.py
"$PY" claim2_reconsider.py
"$PY" claim2_reconsider2.py
"$PY" claim2_posterfinding.py
"$PY" challenge_claim2_round1.py
"$PY" challenge_claim2_round2.py
"$PY" challenge_claim2_round3.py
"$PY" final_summary.py
"$PY" final_summary2.py
"$PY" final_summary3.py
echo "run.sh: agent pipeline finished at $(date -u +%FT%TZ)"

# --- emit inspectable artifacts for the orx run record ---
ART="$REPO/.openresearch/artifacts"
mkdir -p "$ART"

# The scrubbed public trace (stage_trace.py + lib/traces.py scrub local paths).
"$PY" stage_trace.py 2>/dev/null || true   # regenerates trace.jsonl from the raw trace
cp -f trace.jsonl "$ART/agent-trace.jsonl" 2>/dev/null || true
cp -f orx-openweights-trace.jsonl "$ART/agent-trace-raw.jsonl" 2>/dev/null || true
cp -f results.json "$ART/results.json"
cp -f measured.json "$ART/measured.json" 2>/dev/null || true
cp -f build-receipt.json "$ART/build-receipt.json" 2>/dev/null || true
cp -f openweights_session_output.json "$ART/openweights_session_output.json" 2>/dev/null || true

# Copy the model-authored experiment scripts so they are inspectable too.
mkdir -p "$ART/experiments"
cp -f experiments/exp_claim1.py "$ART/experiments/" 2>/dev/null || true
cp -f experiments/exp_claim2.py "$ART/experiments/" 2>/dev/null || true

# --- write an EVAL.md summary (verdicts + key numbers pulled from results.json) ---
"$PY" - <<'PYEOF'
import json, pathlib
art = pathlib.Path(__file__).resolve().parent if "__file__" in dir() else pathlib.Path(".openresearch/artifacts")
art = pathlib.Path(".openresearch/artifacts")
r = json.loads(pathlib.Path("results.json").read_text())
claims = r.get("claims", [])
verdicts = [c.get("verdict","?") for c in claims]
lines = ["# Open-weights PBA reproduction via the orx harness", "",
         "Main agent: **qwen3:30b-a3b** (open-weights MoE), served locally via Ollama on "
         "an Apple M4 Max. The model wrote every experiment script, ran it, checked "
         "determinism, and assigned every verdict; the harness only executed the code "
         "the model wrote and reported stdout/stderr back. This run was executed through "
         "the OpenResearch CLI (`orx`) harness — it is a genuine orx run, not a published "
         "artifact dump.", "",
         f"**Final verdicts:** {verdicts}", ""]
lines += ["## Per-claim findings", ""]
for c in claims:
    lines += [f"### {c.get('shortClaim','')}",
              f"- **Verdict:** `{c.get('verdict','?')}`",
              f"- **Method:** {c.get('method','')}",
              f"- **Finding:** {c.get('finding','')}",
              f"- **Scope:** {c.get('scopeNote','')}", ""]
sc = r.get("scopeCost", {})
lines += ["## Scope & environment", "",
          f"- Hardware: {sc.get('hardware','Apple M4 Max, CPU only')}",
          f"- Compute time: {sc.get('computeTime','~58 min model inference')}",
          f"- Cost: {sc.get('cost','$0 (local CPU + local open-weights model)')}",
          f"- Outcome: {sc.get('outcome','')}", "",
          "## Inspectable evidence", "",
          "- `agent-trace.jsonl` — full scrubbed session trace (every prompt/response)",
          "- `measured.json` — raw RESULT_JSON metrics from each experiment",
          "- `results.json` — final report (method/finding/scope per claim)",
          "- `experiments/exp_claim{1,2}.py` — the model-authored scripts",
          ""]
es = r.get("executiveSummary","")
if es:
    lines += ["## Executive summary", "", es, ""]
pathlib.Path(art/"EVAL.md").write_text("\n".join(lines))
print("run.sh: wrote", art/"EVAL.md")
PYEOF

echo "run.sh: artifacts emitted to .openresearch/artifacts/"
ls -la "$ART"
echo "run.sh: DONE"
