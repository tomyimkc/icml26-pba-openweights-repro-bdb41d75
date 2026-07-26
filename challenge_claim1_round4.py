#!/usr/bin/env python3
"""
Harness-initiated SCRUTINY round 4 (FINAL) for Claim 1 of the Zlw4Kl5HEF repro.

Same honesty rules as rounds 1-3: harness only, no verdict decided here. Round
3's exact (non-discretized) piecewise belief implementation was the right fix
for the round-2 discretization floor, and the model correctly, honestly
downgraded to `inconclusive` when its own positive control (p=1.0) failed to
show clean convergence (measured 0.9938, not ~0.5). Root cause identified: the
script initializes `x0 = 0.5 + 1e-10`, a leftover from the round-2 discretized
grid (chosen there only to avoid an exact tie at a bin boundary). With the
EXACT representation, the very first query point is exactly 0.5 (the median of
a uniform prior on [0,1]), so the initial error is already ~1e-10 -- at the
noise floor of float64 arithmetic -- leaving nothing but floating-point noise
to "converge" over for the remaining 999 steps. This is a narrow, precisely
diagnosed bug (root placed pathologically close to the algorithm's own first
query point), not a new conceptual issue. This round hands that diagnosis to
the model as the FINAL scrutiny pass and lets it decide the final verdict.

Run: ~/.venvs/icml26/bin/python challenge_claim1_round4.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("dow", PAPER_DIR / "driver_openweights.py")
dow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dow)

CLAIM_IDX = 1
SCRIPT_NAME = "exp_claim1.py"
SCRIPT_PATH = PAPER_DIR / "experiments" / SCRIPT_NAME
SESSION_OUT_PATH = PAPER_DIR / "openweights_session_output.json"

MAX_ATTEMPTS = 4


def main() -> int:
    session = json.loads(SESSION_OUT_PATH.read_text())
    claim1_result = session["claim_results"][0]

    round3_script = SCRIPT_PATH.read_text()

    dow.log_event({
        "type": "harness_note",
        "claim": CLAIM_IDX,
        "note": (
            "SCRUTINY ROUND 4 -- FINAL (post-hoc, harness/reviewer-initiated). Root "
            "cause of round 3's failed positive control identified precisely: "
            "`x0 = 0.5 + 1e-10` combined with the exact belief representation's FIRST "
            "query always being exactly 0.5 (median of the uniform prior) means the "
            "initial error is ~1e-10 -- already at float64 noise level -- before any "
            "convergence dynamics can be observed over the other 999 steps. The 1e-10 "
            "offset was a leftover from round 2's discretized grid (chosen to dodge a "
            "bin-boundary tie) and is actively harmful for the exact representation. "
            "Handing this precise, narrow diagnosis to the model as the FINAL scrutiny "
            "round and letting it decide whether to fix it and what the final verdict "
            "is -- no further rounds planned after this regardless of outcome; "
            "inconclusive/toy remains an honest, acceptable result if the issue is "
            "not this, or if scrutiny should reasonably stop here."
        ),
    })

    system_prompt = dow.SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    challenge_prompt = f"""This is a FOURTH and FINAL round of scrutiny on Claim 1. You have made real, \
substantive progress across three rounds: fixing the oracle noise model (round 1), fixing hard-cut \
bisection to a genuine Bayesian belief update (round 2), and fixing the discretization floor with an \
exact piecewise representation (round 3) -- and you honestly downgraded to `inconclusive` in round 3 \
when your own positive control did not look clean. That was the right call given what you knew. One \
more narrow, precisely-identified issue has been found; after this round, no further scrutiny rounds \
are planned regardless of the outcome -- `inconclusive` or `toy` remain completely acceptable final \
answers if this does not resolve cleanly.

CLAIM 1 (verbatim): "{claim1_result['claim']}"

Here is your current experiments/exp_claim1.py (after round 3):

```python
{round3_script}
```

Your current verdict and finding:
  verdict: {claim1_result['verdict']}
  finding: {claim1_result['finding']}
  measured: {json.dumps(claim1_result['measuredMetrics'])}

THE ISSUE: your script sets `x0 = 0.5 + 1e-10` (a leftover from round 2, where it was used to dodge an \
exact tie at a discretized grid boundary -- it is not needed with the exact representation and is now \
actively harmful). With the EXACT piecewise belief representation, the very FIRST query point is \
always exactly the median of the uniform prior on [0, 1], i.e. exactly 0.5. Since x0 is only 1e-10 \
away from 0.5, your very first measured error is already ~1e-10 -- at the numerical noise floor of \
float64 arithmetic (relative machine epsilon ~2.2e-16, but accumulated through many multiply/divide/ \
renormalize operations per step, 1e-10 is close enough to that floor that ratios of successive tiny \
errors become dominated by floating-point noise rather than the algorithm's real convergence \
behavior). That would fully explain why even your noiseless (p=1.0) positive control failed to show \
clean ~0.5 convergence: there is essentially no real "distance left to close" for the algorithm to \
demonstrate geometric convergence over.

The straightforward fix: place x0 somewhere that is NOT close to 0.5 or to any other value your \
algorithm's early queries are likely to land on exactly (e.g. an irrational-looking value well away \
from simple binary fractions, such as 0.37, 0.618, or similar) so there is a real, numerically \
meaningful distance for the algorithm to close over many steps, and the ratio calculation is not \
dominated by floating-point noise.

Please:
1. Decide for yourself whether this diagnosis is correct, and whether adjusting x0 (or any other \
fix you judge necessary) resolves it -- do not just defer to the reviewer.
2. If you agree, make the fix (at minimum, moving x0 away from any value your algorithm's queries \
would land on early) and re-verify: does your noiseless (p=1.0) positive control NOW show clean, \
strong convergence? Keep everything else from your round-3 exact implementation that was working \
(the exact piecewise belief representation, the fixed-p oracle, T as large as the time budget allows).
3. If you disagree, explain concretely why the 1e-10 offset is not the cause, and keep the script \
unchanged.
4. Either way, after your reasoning, give the COMPLETE current script (revised or unchanged) in a \
single fenced ```python code block, self-contained and runnable as-is.
"""
    messages.append({"role": "user", "content": challenge_prompt})

    ok = False
    result_payload = None
    stdout_full = None
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        thinking, content = dow.call_model(messages, purpose=f"claim1_challenge4_attempt{attempt}")
        messages.append({"role": "assistant", "content": content})
        code = dow.extract_code(content)
        if not code:
            last_error = "No fenced python code block found in response."
            messages.append({"role": "user", "content": (
                "I could not find a fenced ```python code block in your last reply. Please "
                "reply again with the FULL current script (revised or unchanged) in a single "
                "```python block."
            )})
            dow.log_event({"type": "harness_note", "claim": CLAIM_IDX, "attempt": attempt,
                            "note": "challenge4: no code block extracted"})
            continue

        SCRIPT_PATH.write_text(code)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "wrote_challenge4_revision", "path": str(SCRIPT_PATH),
                        "bytes": len(code)})

        try:
            proc1 = dow.run_script(SCRIPT_PATH)
        except Exception as e:
            last_error = f"Run failed/timed out: {e}"
            messages.append({"role": "user", "content": (
                f"Your script did not finish within the time budget ({e}). Please reduce "
                "T if needed while keeping the exact (non-discretized) representation, and "
                "reply with the FULL corrected script in a single ```python block."
            )})
            continue

        ok1, msg1, payload1 = dow.check_run(proc1)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "challenge4_run1_result", "ok": ok1, "message": msg1,
                        "stdout_tail": proc1.stdout[-2000:], "stderr_tail": proc1.stderr[-2000:]})
        if not ok1:
            last_error = msg1
            messages.append({"role": "user", "content": (
                "I ran your script exactly as written. It failed:\n\n" + msg1 +
                "\n\nPlease fix it and reply with the FULL corrected script in a single "
                "```python block."
            )})
            continue

        proc2 = dow.run_script(SCRIPT_PATH)
        ok2, msg2, payload2 = dow.check_run(proc2)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "challenge4_run2_result", "ok": ok2, "message": msg2})
        if not ok2 or payload1 != payload2:
            last_error = f"Non-deterministic or failed second run: {msg2}"
            messages.append({"role": "user", "content": (
                "Your script's two runs did not match / the second run failed:\n"
                f"  run1: {json.dumps(payload1)}\n  run2 msg: {msg2}\n"
                "This must be fully deterministic (seeded). Please fix and reply with the "
                "FULL corrected script in a single ```python block."
            )})
            continue

        ok = True
        result_payload = payload1
        stdout_full = proc1.stdout
        print(f"challenge round 4 revision for claim 1 succeeded on attempt {attempt}", file=sys.stderr)
        break

    if not ok:
        print(f"challenge round 4 FAILED after {MAX_ATTEMPTS} attempts: {last_error}", file=sys.stderr)
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": f"challenge round 4 gave up after {MAX_ATTEMPTS} attempts",
                        "last_error": last_error})
        return 1

    messages.append({"role": "user", "content": (
        "Your (possibly revised) script ran successfully twice, deterministically. Here is "
        "exactly what it printed:\n\n--- stdout ---\n" + stdout_full + "\n--- end stdout ---\n\n"
        "This is the FINAL round -- give your FINAL, reconsidered verdict for Claim 1, taking "
        "into account all FOUR rounds of scrutiny and this new evidence. Look closely at whether "
        "your p=1.0 (noiseless) positive control NOW shows clean, strong convergence -- if it "
        "does, you can trust a falsified/verified/toy verdict for the noisy cases; if it still "
        "does not, the honest verdict remains inconclusive (or toy, if partial signal exists), and "
        "that is a completely fine final answer. Follow the hard honesty rules: every number you "
        "write in prose must trace to the RESULT_JSON dict above, to the claim text, or to a "
        "literal constant in your script. Explicitly summarize in your finding how your verdict "
        "evolved across all four rounds and why you are confident in this final answer. Reply with "
        "a JSON object in a single ```json block with EXACTLY these keys:\n"
        '  "verdict": one of "verified", "falsified", "toy", "inconclusive"\n'
        '  "method": what you implemented and how it targets this claim (setting, scale, '
        'seeds, control)\n'
        '  "finding": what the numbers show, in prose, referencing the measured values, and '
        'the full evolution of your verdict across all four scrutiny rounds\n'
        '  "posterFinding": <=200 chars, poster-card summary\n'
        '  "scopeNote": what about the claim you did NOT cover\n'
        '  "shortClaim": <=60 chars, page title for this claim'
    )})
    thinking, content = dow.call_model(messages, purpose="claim1_challenge4_verdict")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": "challenge4 verdict JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the six keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="claim1_challenge4_verdict_retry")
        messages.append({"role": "assistant", "content": content})
        fields = dow.extract_json(content) or {}

    verdict = str(fields.get("verdict", "inconclusive")).lower()
    new_result = {
        "claim": claim1_result["claim"],
        "verdict": verdict,
        "method": fields.get("method", ""),
        "finding": fields.get("finding", ""),
        "posterFinding": (fields.get("posterFinding", "") or "")[:200],
        "scopeNote": fields.get("scopeNote", ""),
        "shortClaim": (fields.get("shortClaim", "") or claim1_result["claim"][:60])[:60],
        "script": SCRIPT_NAME,
        "measuredMetrics": result_payload,
        "runtime1_s": None,
        "verdictAfterRound3": claim1_result["verdict"],
        "measuredMetricsAfterRound3": claim1_result["measuredMetrics"],
    }

    session["claim_results"][0] = new_result
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))

    dow.log_event({
        "type": "challenge_round4_end",
        "claim": CLAIM_IDX,
        "verdict_after_round3": claim1_result["verdict"],
        "final_verdict": verdict,
        "verdict_changed_in_round4": verdict != claim1_result["verdict"],
    })

    print(json.dumps({
        "verdict_after_round3": claim1_result["verdict"],
        "final_verdict": verdict,
        "measured": result_payload,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
