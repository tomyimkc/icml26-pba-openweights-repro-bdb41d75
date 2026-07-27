#!/usr/bin/env python3
"""
Harness-initiated SCRUTINY round 3 for Claim 2 of Zlw4Kl5HEF.

Round 2 fixed the round-1 stall (root placement + grid resolution) and
switched to a two-point geometric decay rate, reaching verdict=verified from
r_0.99_mean=0.9896, r_0.51_mean=0.9998, r_0.50_mean=1.0, r_0.49_mean=0.9991
(3 seeds each). Before trusting that verdict, the harness found two more
things worth putting in front of the model:

1. Claim 2 (verbatim) asserts TWO things jointly: "PBA queries oscillate
   around the truth but steadily draw closer, yielding an estimator that
   rapidly concentrates on the truth." Round 2's script measures ONLY the
   concentration/decay-rate half. It does not measure oscillation at all --
   and the model's own scopeNote for round 2 says exactly that: "Did not
   directly measure query oscillation (only inferred from decay rate <1)".
   Round 1's script *did* measure oscillation directly (sign_changes of
   x_t - x_true over time); round 2 dropped that metric while fixing the
   stall bug. A claim with two asserted dynamical properties where only one
   is measured is a real scope gap, by the model's own admission.

2. The harness independently re-ran round 2's exact update logic (unmodified)
   for p=0.99 across the same 3 seeds and found the reported std=0.0 has an
   explanation worth surfacing: in ALL THREE seeds, the estimate reaches the
   grid-resolution floor (error ~3.7e-6, apparently the closest representable
   grid point to x_true=0.37 on the N=100000 grid) by step 15 of 1000, and
   then never moves again for the remaining ~985 steps (same "frozen query"
   dynamic as round 1's bug, just landing on the CORRECT point this time
   because p=0.99 rarely produces a wrong early response). So the r_0.99
   decay rate, computed as (error[999]/error[0])**(1/999), is being diluted
   by ~985 steps of flat, uninformative floor-sitting -- it understates how
   fast the algorithm actually converges, and calling it a "1000-step
   geometric rate" in prose would be misleading about what was actually
   measured. (p=0.51 and p=0.49 do NOT show this -- their errors keep
   changing throughout all 1000 steps, confirmed by the harness the same
   way.)

This round hands the model both findings (with the harness's own numbers so
it can verify independently) and asks it to decide what, if anything, to add
or change, and what the final verdict is. The harness decides nothing.

Run: /Users/tom/.venvs/icml26/bin/python challenge_claim2_round3.py
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

CLAIM_IDX = 2
SCRIPT_NAME = "exp_claim2.py"
SCRIPT_PATH = PAPER_DIR / "experiments" / SCRIPT_NAME
SESSION_OUT_PATH = PAPER_DIR / "openweights_session_output.json"
RESULTS_PATH = PAPER_DIR / "results.json"

MAX_ATTEMPTS = 6


def main() -> int:
    session = json.loads(SESSION_OUT_PATH.read_text())
    claim2_result = session["claim_results"][1]
    claim2_text = claim2_result["claim"]
    round2_script = SCRIPT_PATH.read_text()

    harness_note = (
        "SCRUTINY ROUND 3 for Claim 2 (harness/reviewer-initiated). Two findings before "
        "trusting round 2's 'verified' verdict: (1) Claim 2 asserts BOTH oscillation AND "
        "concentration; round 2's script measures only concentration (decay rate), and its own "
        "scopeNote admits oscillation was not directly measured. (2) Independent harness rerun of "
        "round 2's exact update logic for p=0.99 shows all 3 seeds hit the grid-resolution floor "
        "(error~3.7e-6) by step 15/1000 and then stay frozen for ~985 steps, so the reported decay "
        "rate is diluted by mostly-flat floor-sitting data (p=0.51/p=0.49 do not show this -- their "
        "errors keep evolving through all 1000 steps, harness-verified). Handing both findings, "
        "with the harness's own reproduced numbers, to the model; it decides what to add/change "
        "and the final verdict."
    )
    dow.log_event({"type": "harness_note", "claim": CLAIM_IDX, "note": harness_note})

    system_prompt = dow.SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    challenge_prompt = f"""This is a THIRD round of scrutiny on Claim 2. Round 1 replaced the hard-cut \
script with a genuine Bayesian PBA; round 2 fixed a root-placement/grid-resolution stall and switched to \
a geometric decay-rate metric, reaching verdict=verified. This round is about whether that verdict \
covers the WHOLE claim and whether the specific numbers can be trusted at face value.

CLAIM 2 (verbatim): "{claim2_text}"

Here is your current experiments/exp_claim2.py (round 2's version, the one behind your 'verified' verdict):

```python
{round2_script}
```

Your round-2 verdict and finding:
  verdict: {claim2_result['verdict']}
  finding: {claim2_result['finding']}
  scopeNote: {claim2_result['scopeNote']}
  measured: {json.dumps(claim2_result['measuredMetrics'])}

TWO THINGS TO CONSIDER:

1. OSCILLATION IS HALF THE CLAIM, AND IT IS NOT MEASURED. Claim 2 asserts TWO things jointly: PBA \
queries "oscillate around the truth" AND "steadily draw closer" / "rapidly concentrate." Your round-2 \
script measures only the concentration half (the decay rate r). Your own scopeNote says so directly: \
"Did not directly measure query oscillation (only inferred from decay rate <1)." Round 1's script *did* \
measure oscillation directly (counting sign changes of x_t - x_true over time) before it was dropped \
while fixing the stall bug in round 2. Per AGENT_BRIEF, 'verified' requires that what you tested is *the \
thing the claim asserts* -- a two-part dynamical claim with only one part measured is a real scope gap by \
your own admission, not a nitpick.

2. THE p=0.99 STD=0.0 HAS AN EXPLANATION WORTH YOUR OWN VERIFICATION. The harness independently re-ran \
your round-2 update logic (unmodified) for p=0.99 across the same 3 seeds (0, 1, 2) and found: in ALL \
THREE seeds, the estimate reaches the grid-resolution floor (error~3.7e-6, apparently the nearest \
representable grid point to x_true=0.37 on your N=100000 grid) by step 15 of 1000, and then the query \
never moves again for the remaining ~985 steps. So your r_0.99 decay rate, computed as \
(error[999]/error[0])**(1/999) over the full 1000 steps, is diluted by ~985 steps of flat, uninformative \
floor-sitting -- it understates how fast the algorithm actually converges, and describing it in prose as \
a "1000-step" rate could be misleading about what those 1000 steps actually contain. By contrast, the \
harness verified p=0.51 and p=0.49 do NOT hit this floor -- their errors keep evolving throughout all \
1000 steps in both cases. You do not have to trust the harness's numbers blindly -- verify this yourself \
(e.g. by printing/inspecting the error trajectory for p=0.99 in your own script) before deciding whether \
and how to address it.

Please decide for yourself:
- Whether to add a genuine oscillation measurement (e.g. sign changes of x_t - x_true over time, as round \
1 did, or another metric of your choosing) to this script so BOTH halves of the claim are actually \
tested, or to explain concretely why decay rate alone is sufficient evidence for 'verified' on the full \
claim text.
- Whether/how to address the p=0.99 floor dilution (e.g. a shorter T for the fast-converging case, a \
metric less sensitive to post-floor flatness, more/different p values, or simply an honest note in your \
scopeNote about what the 1000-step number actually reflects for high p) -- your call.
- If you disagree with either point, say so concretely and you may leave the script unchanged.

Either way, give the COMPLETE current script (revised or unchanged) in a single fenced ```python code \
block, self-contained, deterministic (seed any seed-generation too), and runnable well within the \
5-minute budget (your current script runs in about 4.5 seconds, so you have a lot of headroom).
"""
    messages.append({"role": "user", "content": challenge_prompt})

    ok = False
    result_payload = None
    stdout_full = None
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        thinking, content = dow.call_model(messages, purpose=f"claim2_challenge3_attempt{attempt}")
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
                            "note": "challenge3: no code block extracted"})
            continue

        SCRIPT_PATH.write_text(code)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "wrote_challenge3_revision", "path": str(SCRIPT_PATH),
                        "bytes": len(code)})

        try:
            proc1 = dow.run_script(SCRIPT_PATH)
        except Exception as e:
            last_error = f"Run failed/timed out: {e}"
            dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                            "action": "run1_timeout_or_error", "error": str(e)})
            messages.append({"role": "user", "content": (
                f"Your script did not finish within the time budget or errored ({e}). Please "
                "reduce scale if needed and reply with the FULL corrected script in a single "
                "```python block."
            )})
            continue

        ok1, msg1, payload1 = dow.check_run(proc1)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "challenge3_run1_result", "ok": ok1, "message": msg1,
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
                        "action": "challenge3_run2_result", "ok": ok2, "message": msg2})
        if not ok2 or payload1 != payload2:
            last_error = f"Non-deterministic or failed second run: {msg2}"
            dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                            "action": "determinism_mismatch", "run1": payload1, "run2_msg": msg2})
            messages.append({"role": "user", "content": (
                "Your script's two runs did not match / the second run failed:\n"
                f"  run1: {json.dumps(payload1)}\n  run2 msg: {msg2}\n"
                "This must be fully deterministic. Please fix and reply with the FULL corrected "
                "script in a single ```python block."
            )})
            continue

        ok = True
        result_payload = payload1
        stdout_full = proc1.stdout
        print(f"challenge round 3 for claim 2 succeeded on attempt {attempt}", file=sys.stderr)
        break

    if not ok:
        print(f"challenge round 3 for claim 2 FAILED after {MAX_ATTEMPTS} attempts: {last_error}",
              file=sys.stderr)
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": f"challenge round 3 gave up after {MAX_ATTEMPTS} attempts",
                        "last_error": last_error})
        return 1

    messages.append({"role": "user", "content": (
        "Your (possibly revised) script ran successfully twice, deterministically. Here is exactly "
        "what it printed:\n\n--- stdout ---\n" + stdout_full + "\n--- end stdout ---\n\n"
        "This is the FINAL round for Claim 2 -- give your FINAL, reconsidered verdict, taking into "
        "account all three rounds (hard-cut -> Bayesian rewrite -> stall/metric fix -> "
        "oscillation/floor-dilution review). Follow the hard honesty rules: "
        "verified/falsified/toy/inconclusive, and every number you write in prose must trace to the "
        "RESULT_JSON dict above, to the claim text, or to a literal constant in your script. If your "
        "evidence covers only part of the two-part claim, or the floor-dilution issue was not fully "
        "resolved, toy or inconclusive is a completely fine, honest final answer -- do not inflate it "
        "to verified just because a subset of the claim held up. Summarize the full evolution across "
        "all three rounds in your finding. Reply with a JSON object in a single ```json block with "
        "EXACTLY these keys:\n"
        '  "verdict": one of "verified", "falsified", "toy", "inconclusive"\n'
        '  "method": what you implemented and how it targets this claim (representation, scale, '
        'seeds, control, and whether it covers oscillation, concentration, or both)\n'
        '  "finding": what the numbers show, in prose, referencing the measured values, and the '
        'evolution across all three rounds\n'
        '  "posterFinding": <=200 chars, poster-card summary\n'
        '  "scopeNote": honest final scope note -- what about the claim you did NOT cover\n'
        '  "shortClaim": <=60 chars, page title for this claim'
    )})
    thinking, content = dow.call_model(messages, purpose="claim2_challenge3_verdict")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": "challenge3 verdict JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the six keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="claim2_challenge3_verdict_retry")
        messages.append({"role": "assistant", "content": content})
        fields = dow.extract_json(content) or {}

    verdict = str(fields.get("verdict", "inconclusive")).lower()
    new_result = {
        "claim": claim2_text,
        "verdict": verdict,
        "method": fields.get("method", ""),
        "finding": fields.get("finding", ""),
        "posterFinding": (fields.get("posterFinding", "") or "")[:200],
        "scopeNote": fields.get("scopeNote", ""),
        "shortClaim": (fields.get("shortClaim", "") or claim2_text[:60])[:60],
        "script": SCRIPT_NAME,
        "measuredMetrics": result_payload,
        "runtime1_s": None,
        "verdictAfterRound2": claim2_result["verdict"],
        "measuredMetricsAfterRound2": claim2_result["measuredMetrics"],
    }

    session["claim_results"][1] = new_result
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))

    results = json.loads(RESULTS_PATH.read_text())
    c2 = results["claims"][1]
    c2["verdict"] = verdict
    c2["method"] = new_result["method"]
    c2["finding"] = new_result["finding"]
    c2["posterFinding"] = new_result["posterFinding"]
    c2["scopeNote"] = new_result["scopeNote"]
    c2["shortClaim"] = new_result["shortClaim"] or c2.get("shortClaim", "")
    c2["script"] = SCRIPT_NAME
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")

    dow.log_event({
        "type": "challenge_claim2_round3_end",
        "claim": CLAIM_IDX,
        "verdict_after_round2": claim2_result["verdict"],
        "final_verdict": verdict,
        "verdict_changed": verdict != claim2_result["verdict"],
    })

    print(json.dumps({
        "verdict_after_round2": claim2_result["verdict"],
        "final_verdict": verdict,
        "measured": result_payload,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
