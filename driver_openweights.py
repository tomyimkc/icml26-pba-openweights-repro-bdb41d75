#!/usr/bin/env python3
"""
Harness driver for the OpenResearch Open-Weights Award reproduction of
ICML 2026 paper Zlw4Kl5HEF ("Probabilistic Bisection Algorithm Provably
Achieves Exponential Convergence").

HONESTY NOTE (read this before touching the code below):
This script is harness/plumbing ONLY. It does not decide anything about the
science. Its only jobs are:
  1. Send the open-weights model (qwen3:30b-a3b, local via Ollama) the paper
     claim + the experiment-script contract from AGENT_BRIEF.md.
  2. Extract the python code block the model writes and execute it verbatim.
  3. Feed stdout/stderr back to the model on failure and let IT decide the fix.
  4. Ask the model to interpret its own measured RESULT_JSON and assign the
     verdict + write the prose. The harness never assigns a verdict, never
     invents a number, and never edits the model's code beyond saving it to
     disk.
  5. Log every prompt and every response (including the model's own chain of
     thought, where the API exposes it) to orx-openweights-trace.jsonl.

Run: ~/.venvs/icml26/bin/python driver_openweights.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
TRACE_PATH = PAPER_DIR / "orx-openweights-trace.jsonl"
MODEL = "qwen3:30b-a3b"
OLLAMA_URL = "http://localhost:11434/api/chat"
PYTHON_BIN = "~/.venvs/icml26/bin/python"  # same interpreter build_logbook.py uses
NUM_CTX = 24576
MAX_ATTEMPTS_PER_CLAIM = 6
SCRIPT_TIMEOUT_S = 400  # contract says "under 5 minutes"; leave slack for a slow CPU run

paper = json.loads((PAPER_DIR / "paper.json").read_text())

CONTRACT = """\
## Experiment script contract (verbatim from AGENT_BRIEF.md — you must follow this exactly)

Each script must:
- Use only the Python standard library, numpy, scipy, pandas, matplotlib (no
  network, no model downloads, no GPU, no torch).
- Be fully deterministic: seed everything (np.random.default_rng(0)). The
  script will be executed twice and the run is rejected if the two runs
  disagree.
- Finish in under 5 minutes on a laptop CPU.
- Implement the paper's own construction independently from its description
  (there is no repo to import here — you must implement PBA yourself from
  what you know of it).
- Run at the largest scale that fits the time budget, not a micro-toy. Scale
  is what separates `verified` from `toy`.
- Include a control that relaxes the claim's condition and shows the
  property degrades. A property that holds even when its precondition is
  violated is evidence the test is vacuous.
- Print exactly one machine-readable line at the end, of the exact form:
    print("RESULT_JSON " + json.dumps({"metric name": value, ...}))
  Put every quantity a reader needs to judge the claim in that dict, with
  readable label keys. Print human-readable detail above it, but the
  RESULT_JSON line is the ONLY place numbers can be sourced from later —
  anything you want to discuss in prose later must appear in this dict.

## Hard honesty rules
1. Never write a number you did not measure. Every decimal you later quote
   in prose must trace to this RESULT_JSON dict, to the official claim text,
   or to a literal constant in your own script.
2. Never claim a scale you did not run. State the actual n / T / seeds used.
3. A falsification is a full, valuable result — do not fudge a real failure
   toward "verified". But it needs a positive control proving your own
   implementation is correct elsewhere, and it needs to show the deviation
   is outside noise (multiple seeds, report the spread).
4. If you genuinely cannot target the claim, say so honestly later — do not
   fake success.
"""

SYSTEM_PROMPT = f"""You are the lead scientist reproducing one claim of an ICML 2026 paper as \
part of an open-weights-model reproduction. You are qwen3:30b-a3b, running fully \
locally with no network access and no GPU. YOU make every scientific decision: how \
to test the claim, what scale to run at, what control to include, and — later — \
what the numbers mean and what verdict they support. The process around you (a \
harness) only executes the exact code you write and reports back stdout/stderr \
verbatim; it will not edit your code or invent any numbers on your behalf.

Paper title: {paper['title']}
Paper abstract: {paper['abstract']}
(No arXiv id or code repo is available for this paper — you must implement the
probabilistic bisection algorithm (PBA) yourself from your own knowledge of it
and from the claim text below. Say explicitly in your reasoning if you are
uncertain about any part of the construction.)

{CONTRACT}
"""


def log_event(event: dict) -> None:
    event = {"ts": time.time(), **event}
    with TRACE_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")


def call_model(messages: list[dict], purpose: str) -> tuple[str, str]:
    """Call the local open-weights model. Returns (thinking, content)."""
    log_event({"type": "prompt", "purpose": purpose, "model": MODEL, "messages": messages})
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3, "num_ctx": NUM_CTX},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as resp:
        data = json.loads(resp.read())
    dt = time.time() - t0
    msg = data.get("message", {})
    content = msg.get("content", "")
    thinking = msg.get("thinking", "")
    log_event({
        "type": "response",
        "purpose": purpose,
        "model": MODEL,
        "thinking": thinking,
        "content": content,
        "wall_seconds": dt,
        "eval_count": data.get("eval_count"),
        "eval_duration_ns": data.get("eval_duration"),
    })
    print(f"    [{purpose}] model responded in {dt:.1f}s "
          f"({data.get('eval_count')} tokens)", file=sys.stderr)
    return thinking, content


def extract_code(content: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", content, re.S)
    if not blocks:
        return None
    # the model's actual script is virtually always the largest fenced block
    return max(blocks, key=len).strip() + "\n"


def extract_json(content: str) -> dict | None:
    blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", content, re.S)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            continue
    # fall back: maybe the whole content is JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def run_script(script_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON_BIN, str(script_path)],
        cwd=str(PAPER_DIR),
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_S,
    )


RESULT_LINE = re.compile(r"^RESULT_JSON\s+(\{.*\})\s*$", re.M)


def check_run(proc: subprocess.CompletedProcess) -> tuple[bool, str, dict | None]:
    """Returns (ok, message, result_json)."""
    if proc.returncode != 0:
        return False, (
            f"Script exited with code {proc.returncode}.\n"
            f"--- stdout (tail) ---\n{proc.stdout[-3000:]}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-3000:]}"
        ), None
    m = RESULT_LINE.search(proc.stdout or "")
    if not m:
        return False, (
            "Script exited 0 but printed no line starting with 'RESULT_JSON '. "
            "You must print exactly one such line, e.g.\n"
            'print("RESULT_JSON " + json.dumps({"max abs error": err, "n": n}))\n'
            f"--- stdout (tail) ---\n{proc.stdout[-3000:]}"
        ), None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        return False, f"RESULT_JSON line is not valid JSON: {e}", None
    if not isinstance(payload, dict) or not payload:
        return False, "RESULT_JSON must be a non-empty JSON object.", None
    return True, "ok", payload


def drive_claim(idx: int, claim_text: str) -> dict:
    """Runs the full write -> execute -> feedback -> iterate loop for one claim.
    Returns a dict with the model's final verdict + prose + the measured metrics.
    """
    script_name = f"exp_claim{idx}.py"
    script_path = PAPER_DIR / "experiments" / script_name
    print(f"\n=== Claim {idx}: {claim_text[:90]}... ===", file=sys.stderr)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    other_claims = "\n".join(
        f"  {i+1}. {c}" for i, c in enumerate(paper["claims"]) if i + 1 != idx
    )
    user_turn1 = f"""Your assignment: design and write experiments/{script_name} to test THIS \
official claim (verbatim):

  CLAIM {idx}: "{claim_text}"

(For context, the paper's other official claim(s), which you are NOT assigned this \
time, are:\n{other_claims})

First, in 1-3 short paragraphs of plain prose, explain your test design: what you \
will implement, at what scale, what exactly you will measure, and what control you \
will include to relax the claim's precondition. Then give the COMPLETE script in a \
single fenced ```python code block. The whole file must be self-contained and \
runnable as-is."""
    messages.append({"role": "user", "content": user_turn1})

    result_payload = None
    ok = False
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS_PER_CLAIM + 1):
        thinking, content = call_model(messages, purpose=f"claim{idx}_write_attempt{attempt}")
        messages.append({"role": "assistant", "content": content})
        code = extract_code(content)
        if not code:
            last_error = "No fenced python code block found in your response."
            messages.append({"role": "user", "content": (
                "I could not find a fenced ```python code block in your last reply. "
                "Please reply again with the FULL script in a single ```python block."
            )})
            log_event({"type": "harness_note", "claim": idx, "attempt": attempt,
                       "note": "no code block extracted"})
            continue

        script_path.write_text(code)
        log_event({"type": "harness_action", "claim": idx, "attempt": attempt,
                   "action": "wrote_script", "path": str(script_path),
                   "bytes": len(code)})

        try:
            proc1 = run_script(script_path)
        except subprocess.TimeoutExpired:
            last_error = f"Script did not finish within {SCRIPT_TIMEOUT_S}s (timed out)."
            log_event({"type": "harness_action", "claim": idx, "attempt": attempt,
                       "action": "run1_timeout"})
            messages.append({"role": "user", "content": (
                f"Your script did not finish within {SCRIPT_TIMEOUT_S} seconds on this "
                "CPU-only laptop and was killed. Please reduce the scale (smaller n/T/"
                "seeds, or a faster inner loop) while still targeting the claim, and "
                "give the full corrected script in a single ```python block."
            )})
            continue

        ok1, msg1, payload1 = check_run(proc1)
        log_event({"type": "harness_action", "claim": idx, "attempt": attempt,
                   "action": "run1_result", "ok": ok1, "message": msg1,
                   "stdout_tail": proc1.stdout[-2000:], "stderr_tail": proc1.stderr[-2000:]})
        if not ok1:
            last_error = msg1
            messages.append({"role": "user", "content": (
                "I ran your script exactly as written. It failed:\n\n" + msg1 +
                "\n\nPlease fix it and reply with the FULL corrected script in a single "
                "```python block."
            )})
            continue

        # determinism check: run a second time, must match
        try:
            proc2 = run_script(script_path)
        except subprocess.TimeoutExpired:
            last_error = "Second determinism run timed out."
            messages.append({"role": "user", "content": (
                "Your script succeeded once, but a second identical run of the same "
                "file did not finish within the time budget (it must be fast and "
                "deterministic since it will be executed twice). Please make it faster "
                "and reply with the FULL corrected script in a single ```python block."
            )})
            continue
        ok2, msg2, payload2 = check_run(proc2)
        log_event({"type": "harness_action", "claim": idx, "attempt": attempt,
                   "action": "run2_result", "ok": ok2, "message": msg2})
        if not ok2:
            last_error = "Second run failed: " + msg2
            messages.append({"role": "user", "content": (
                "Your script succeeded on the first run but FAILED on a second, "
                "identical run (it must be deterministic — it will always be executed "
                "twice and rejected if the runs disagree):\n\n" + msg2 +
                "\n\nPlease fix it and reply with the FULL corrected script in a single "
                "```python block."
            )})
            continue

        if payload1 != payload2:
            last_error = (
                f"Non-deterministic RESULT_JSON: run1={payload1!r} run2={payload2!r}"
            )
            log_event({"type": "harness_action", "claim": idx, "attempt": attempt,
                       "action": "determinism_mismatch", "run1": payload1, "run2": payload2})
            messages.append({"role": "user", "content": (
                "Your script ran twice and printed DIFFERENT RESULT_JSON values:\n"
                f"  run 1: {json.dumps(payload1)}\n  run 2: {json.dumps(payload2)}\n"
                "This means something is not properly seeded (e.g. an unseeded RNG, "
                "set() iteration order, or wall-clock timing leaking into the metrics). "
                "Please make it fully deterministic and reply with the FULL corrected "
                "script in a single ```python block."
            )})
            continue

        # success
        ok = True
        result_payload = payload1
        stdout_full = proc1.stdout
        print(f"    claim {idx}: script succeeded on attempt {attempt}", file=sys.stderr)
        break

    if not ok:
        log_event({"type": "harness_note", "claim": idx,
                   "note": f"gave up after {MAX_ATTEMPTS_PER_CLAIM} attempts",
                   "last_error": last_error})
        # Let the model itself deliver the honest verdict for a claim it could not get running.
        messages.append({"role": "user", "content": (
            f"We are out of attempts ({MAX_ATTEMPTS_PER_CLAIM}) for this claim's script — "
            "the last error was:\n\n" + str(last_error) + "\n\n"
            "Since no working, deterministic, measured script exists for this claim, the "
            "honest verdict is `inconclusive` (per the honesty rules: 'if you genuinely "
            "cannot target a claim, mark it inconclusive and say why'). Reply with a JSON "
            "object in a ```json block with EXACTLY these keys: verdict (must be "
            '"inconclusive"), method, finding, posterFinding (<=200 chars), scopeNote, '
            "shortClaim (<=60 chars). Explain honestly in scopeNote/finding what went wrong."
        )})
        thinking, content = call_model(messages, purpose=f"claim{idx}_verdict_failed")
        messages.append({"role": "assistant", "content": content})
        fields = extract_json(content) or {}
        return {
            "claim": claim_text,
            "verdict": "inconclusive",
            "method": fields.get("method", "The assigned experiment script never reached a "
                                            "working, deterministic state within the attempt "
                                            "budget; see scopeNote."),
            "finding": fields.get("finding", str(last_error)),
            "posterFinding": fields.get("posterFinding", "Open-weights model could not "
                                                            "produce a working script for this "
                                                            "claim in time.")[:200],
            "scopeNote": fields.get("scopeNote", str(last_error)),
            "shortClaim": fields.get("shortClaim", claim_text[:60]),
            "script": None,
            "measuredMetrics": None,
        }

    # Interpretation turn: give the model its own measured numbers and ask for
    # the verdict + prose. The harness does not compute or suggest a verdict.
    messages.append({"role": "user", "content": (
        "Your script ran successfully (twice, deterministically). Here is exactly what it "
        "printed:\n\n--- stdout ---\n" + stdout_full + "\n--- end stdout ---\n\n"
        "Now interpret these measured numbers against the claim. Follow the hard honesty "
        "rules: verified/falsified/toy/inconclusive, and every number you write in prose "
        "must trace to the RESULT_JSON dict above, to the claim text, or to a literal "
        "constant in your script (nothing else). Consider your control run explicitly. "
        "Reply with a JSON object in a single ```json block with EXACTLY these keys:\n"
        '  "verdict": one of "verified", "falsified", "toy", "inconclusive"\n'
        '  "method": what you implemented and how it targets this claim (setting, scale, '
        'seeds, control)\n'
        '  "finding": what the numbers show, in prose, referencing the measured values\n'
        '  "posterFinding": <=200 chars, poster-card summary\n'
        '  "scopeNote": what about the claim you did NOT cover\n'
        '  "shortClaim": <=60 chars, page title for this claim'
    )})
    thinking, content = call_model(messages, purpose=f"claim{idx}_verdict")
    messages.append({"role": "assistant", "content": content})
    fields = extract_json(content)
    if not fields:
        log_event({"type": "harness_note", "claim": idx,
                   "note": "could not parse verdict JSON; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the six keys requested."
        )})
        thinking, content = call_model(messages, purpose=f"claim{idx}_verdict_retry")
        messages.append({"role": "assistant", "content": content})
        fields = extract_json(content) or {}

    verdict = str(fields.get("verdict", "inconclusive")).lower()
    if verdict not in {"verified", "falsified", "toy", "inconclusive"}:
        log_event({"type": "harness_note", "claim": idx,
                   "note": f"model returned invalid verdict {verdict!r}; recording verbatim, "
                           "will surface as a build error rather than silently coerced"})

    return {
        "claim": claim_text,
        "verdict": verdict,
        "method": fields.get("method", ""),
        "finding": fields.get("finding", ""),
        "posterFinding": (fields.get("posterFinding", "") or "")[:200],
        "scopeNote": fields.get("scopeNote", ""),
        "shortClaim": (fields.get("shortClaim", "") or claim_text[:60])[:60],
        "script": script_name,
        "measuredMetrics": result_payload,
        "runtime1_s": None,  # filled by caller from proc timing if needed
    }


def drive_summary(claim_results: list[dict], total_runtime_s: float) -> dict:
    """Final turn: ask the model to write the paper-level executiveSummary,
    scopeCost.scope/outcome, and conclusion, from its own two claim results."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    summary_block = "\n\n".join(
        f"Claim {i+1}: \"{c['claim']}\"\n"
        f"  verdict: {c['verdict']}\n"
        f"  measured: {json.dumps(c['measuredMetrics'])}\n"
        f"  your finding: {c['finding']}"
        for i, c in enumerate(claim_results)
    )
    messages.append({"role": "user", "content": (
        "You have now completed both claims of this reproduction. Here is a summary of "
        f"what you found:\n\n{summary_block}\n\n"
        f"Total measured CPU runtime across both experiment scripts: {total_runtime_s:.1f}s "
        "on an Apple M4 Max, CPU only, $0 cost (no GPU, no network).\n\n"
        "Write the paper-level summary. Reply with a JSON object in a single ```json block "
        "with EXACTLY these keys:\n"
        '  "executiveSummary": 3-5 sentences, OUTCOME FIRST: what reproduced, what did not, '
        "what exactly was tested vs the paper's full setup, and the hardware/time/cost\n"
        '  "scope": one sentence describing the reduced scope relative to the full paper\n'
        '  "outcome": one short sentence, the bottom-line outcome\n'
        '  "conclusion": a paragraph: which claims were supported/falsified/inconclusive, '
        "plus reproducibility notes\n"
        '  "links": a JSON array of any source URLs you genuinely used or want to cite '
        "(it is fine for this to be an empty array — do not invent a URL you did not use)"
    )})
    thinking, content = call_model(messages, purpose="paper_summary")
    messages.append({"role": "assistant", "content": content})
    fields = extract_json(content) or {}
    return fields


def main() -> int:
    TRACE_PATH.write_text("")  # fresh trace for this run
    log_event({"type": "session_start", "paper_orid": paper["orid"],
               "paper_title": paper["title"], "model": MODEL,
               "role_of_model": "main reproducing agent — writes all experiment code, "
                                 "interprets all results, assigns all verdicts",
               "role_of_harness": "executes exactly what the model writes; records the "
                                   "full trace; performs no scientific judgment"})

    claim_results = []
    t_run_start = time.time()
    for i, claim_text in enumerate(paper["claims"], start=1):
        res = drive_claim(i, claim_text)
        claim_results.append(res)
    total_runtime_s = time.time() - t_run_start

    summary = drive_summary(claim_results, total_runtime_s)

    out = {
        "paper": {"orid": paper["orid"], "title": paper["title"], "arxiv": paper.get("arxiv") or None},
        "claim_results": claim_results,
        "summary": summary,
        "total_runtime_s": total_runtime_s,
    }
    (PAPER_DIR / "openweights_session_output.json").write_text(json.dumps(out, indent=2))
    log_event({"type": "session_end", "claims_completed": len(claim_results),
               "verdicts": [c["verdict"] for c in claim_results],
               "total_runtime_s": total_runtime_s})
    print("\n=== DONE ===", file=sys.stderr)
    print(json.dumps({"verdicts": [c["verdict"] for c in claim_results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
