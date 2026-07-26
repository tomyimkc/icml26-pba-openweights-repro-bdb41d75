#!/usr/bin/env python3
"""
Harness-initiated SCRUTINY round 3 for Claim 1 of the Zlw4Kl5HEF reproduction.

Same honesty rules as rounds 1-2: harness only, no verdict decided here.
Round 2 fixed the algorithm to a genuine Bayesian belief-density update (not a
hard interval cut) but used a FIXED discretization grid of N=10001 bins over
T=100 steps. log2(10001) ~= 13.3, so the belief distribution collapses to a
single grid cell (full resolution) after ~14 steps -- for the remaining ~86 of
the 100 measured steps, no further refinement is even POSSIBLE, and the
"error" floors at the grid-cell width regardless of p. That inflates the
average step-ratio toward 1 for every p, including p=1.0 (noiseless), which is
why even the round-2 "positive control" only showed avg_ratio=0.9416 -- weak,
not the clean ~0.5-ish geometric convergence a correct noiseless positive
control should show. Per the reproduction's own honesty rules, a positive
control that itself does not clearly show the expected behavior makes a
falsification indistinguishable from a bug. This round hands the model an
EXACT (non-discretized) representation: since each Bayesian update multiplies
the belief by a piecewise-constant factor (p or 1-p) split exactly at the
query point x_t, the belief stays exactly piecewise-constant with breakpoints
at the past query points -- so it can be tracked exactly with O(T) pieces
(growing by at most one breakpoint per step), with NO discretization floor,
for however many steps T the time budget allows. This standard technique is
handed to the model as a recipe (not code to paste unread); the model decides
whether to use it, verifies it, and assigns the final verdict.

Run: ~/.venvs/icml26/bin/python challenge_claim1_round3.py
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

    round2_script = SCRIPT_PATH.read_text()

    dow.log_event({
        "type": "harness_note",
        "claim": CLAIM_IDX,
        "note": (
            "SCRUTINY ROUND 3 (post-hoc, harness/reviewer-initiated). Round 2 fixed "
            "the update rule to a genuine Bayesian belief-density update but used a "
            "fixed N=10001-bin grid over T=100 steps. log2(10001)~=13.3, so the "
            "belief collapses to a single grid cell after ~14 steps and cannot refine "
            "further for the remaining ~86 measured steps -- a discretization floor "
            "that inflates avg_ratio toward 1 for EVERY p, including the p=1.0 "
            "positive control, which measured only 0.9416 (weak; clean noiseless "
            "convergence should look much stronger, e.g. close to the ~0.5-0.65 seen "
            "in the round-1/round-2 hard-cut noiseless baselines). Per the reproduction's "
            "own honesty rules, a weak/failing positive control makes a falsification "
            "indistinguishable from a bug. Confirmed the floor-step arithmetic directly: "
            "log2(10001) = 13.29 < T = 100. Handing the model an EXACT, non-discretized "
            "piecewise-constant belief representation (breakpoints at past query points; "
            "grows by <=1 breakpoint per step, so O(T) pieces, no floor) as a recipe, and "
            "letting the model implement/verify it and decide the final verdict."
        ),
    })

    system_prompt = dow.SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    challenge_prompt = f"""This is a THIRD round of scrutiny on Claim 1. Round 1 fixed the oracle to a \
fixed-probability Bernoulli response. Round 2 fixed the update rule from a hard interval cut to a \
genuine Bayesian belief-density update -- good, real progress. But a measurement-scale issue in \
round 2's implementation needs your judgment before this can be reported as a falsification.

CLAIM 1 (verbatim): "{claim1_result['claim']}"

Here is your current experiments/exp_claim1.py (after round 2's fix):

```python
{round2_script}
```

Your current verdict and finding:
  verdict: {claim1_result['verdict']}
  finding: {claim1_result['finding']}
  measured: {json.dumps(claim1_result['measuredMetrics'])}

THE NEW CONCERN: Your script discretizes the belief distribution into N=10001 fixed grid bins and \
runs T=100 steps. log2(10001) ~= 13.3 -- so after only about 14 steps, the belief has already \
collapsed onto a SINGLE grid cell (the finest resolution your grid can represent) and literally \
cannot refine any further for the remaining ~86 of the 100 measured steps, no matter what p is. \
Once that floor is hit, the "error" stops changing (or changes only due to floating-point noise), \
so the step ratio sits at ~1 for the majority of your measurement window -- inflating avg_ratio \
toward 1 regardless of the true underlying convergence rate.

This explains a specific red flag in your own round-2 numbers: even your p=1.0 (perfectly reliable, \
noiseless) positive control only measured avg_ratio=0.9416 -- weak, not the strong, clean \
convergence a correct noiseless run should show. A weak positive control means (per the honesty \
rules you were given) the falsification is not yet distinguishable from an implementation \
artifact -- in this case, almost certainly the discretization floor, not a real property of PBA.

A standard EXACT (non-discretized) fix, since you already understand the Bayesian update rule: the \
belief distribution stays exactly piecewise-constant throughout -- each update multiplies everything \
left of x_t by one factor and everything right of x_t by another, so the ONLY new "breakpoint" \
introduced at each step is x_t itself. That means you can track the belief EXACTLY as a sorted list \
of breakpoints (starting as just [0, 1], one piece) plus a weight per piece, with the list growing \
by AT MOST one breakpoint per step -- so after T steps there are at most T+2 breakpoints, never more, \
and there is NO discretization floor no matter how large T is. Sketch:

  breakpoints = [0.0, 1.0]   # sorted boundaries; piece i spans breakpoints[i]..breakpoints[i+1]
  weights     = [1.0]        # unnormalized height of each piece (piece MASS = height * width)

  for step in range(T):
      # total mass and the point x_t where cumulative mass first reaches half of total
      # (walk the pieces in order, accumulating height*width, and interpolate within
      #  the piece where the running total crosses total_mass/2)
      # ... compute x_t ...

      # record error = abs(x_t - x0)

      # noisy response: correct with prob p, independent of |x_t - x0|

      # if x_t is not already a breakpoint, insert it (splitting the piece that
      # contained it into two pieces that both start with the SAME height as the
      # original piece -- you are only adding a boundary, not changing the belief yet)

      # THEN multiply every piece with its right edge <= x_t by one factor (p or 1-p,
      # depending on the response direction) and every piece with its left edge >= x_t
      # by the other factor, then renormalize all weights so total mass is 1 again

Please:
1. Decide for yourself whether this concern is valid, and whether the exact piecewise approach (or \
some other fix that avoids a resolution floor, e.g. a scipy/analytic approach if you prefer) is the \
right response -- do not just defer to the reviewer.
2. If you agree, rewrite experiments/exp_claim1.py using an EXACT belief representation with no \
discretization floor, at the largest T the time budget allows. Verify your positive control (p=1.0, \
noiseless) shows CLEAN, strong convergence before trusting any falsification at p=0.9. Keep the \
p close to 0.5 control that should show markedly worse convergence.
3. If you disagree, explain concretely and specifically why the N=10001/T=100 discretization does \
NOT create a resolution floor that dominates your measurement window, with the arithmetic to back \
it up, and keep the script unchanged.
4. Either way, after your reasoning, give the COMPLETE current script (revised or unchanged) in a \
single fenced ```python code block, self-contained and runnable as-is.
"""
    messages.append({"role": "user", "content": challenge_prompt})

    ok = False
    result_payload = None
    stdout_full = None
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        thinking, content = dow.call_model(messages, purpose=f"claim1_challenge3_attempt{attempt}")
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
            messages.append({"role": "user", "content": (
                f"Your script did not finish within the time budget ({e}). The exact "
                "piecewise approach should be fast (O(T^2)) for T up to a few hundred -- "
                "please reduce T if needed while keeping it exact (no fixed grid), and "
                "reply with the FULL corrected script in a single ```python block."
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
        print(f"challenge round 3 revision for claim 1 succeeded on attempt {attempt}", file=sys.stderr)
        break

    if not ok:
        print(f"challenge round 3 FAILED after {MAX_ATTEMPTS} attempts: {last_error}", file=sys.stderr)
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": f"challenge round 3 gave up after {MAX_ATTEMPTS} attempts",
                        "last_error": last_error})
        return 1

    messages.append({"role": "user", "content": (
        "Your (possibly revised) script ran successfully twice, deterministically. Here is "
        "exactly what it printed:\n\n--- stdout ---\n" + stdout_full + "\n--- end stdout ---\n\n"
        "Now give your FINAL, reconsidered verdict for Claim 1, taking into account all THREE "
        "rounds of scrutiny (oracle noise model, hard-cut vs Bayesian update, and discretization "
        "floor) and this new evidence. Pay special attention to whether your p=1.0 (noiseless) "
        "positive control NOW shows clean, strong convergence -- if it still does not, say so "
        "honestly rather than reporting a falsification. Follow the hard honesty rules: "
        "verified/falsified/toy/inconclusive, and every number you write in prose must trace "
        "to the RESULT_JSON dict above, to the claim text, or to a literal constant in your "
        "script. Explicitly say in your finding how/why your verdict evolved (or did not) across "
        "all three rounds. Reply with a JSON object in a single ```json block with EXACTLY these "
        "keys:\n"
        '  "verdict": one of "verified", "falsified", "toy", "inconclusive"\n'
        '  "method": what you implemented and how it targets this claim (setting, scale, '
        'seeds, control)\n'
        '  "finding": what the numbers show, in prose, referencing the measured values, and '
        'how your verdict evolved across all three scrutiny rounds\n'
        '  "posterFinding": <=200 chars, poster-card summary\n'
        '  "scopeNote": what about the claim you did NOT cover\n'
        '  "shortClaim": <=60 chars, page title for this claim'
    )})
    thinking, content = dow.call_model(messages, purpose="claim1_challenge3_verdict")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": "challenge3 verdict JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the six keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="claim1_challenge3_verdict_retry")
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
        "verdictAfterRound2": claim1_result["verdict"],
        "measuredMetricsAfterRound2": claim1_result["measuredMetrics"],
    }

    session["claim_results"][0] = new_result
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))

    dow.log_event({
        "type": "challenge_round3_end",
        "claim": CLAIM_IDX,
        "verdict_after_round2": claim1_result["verdict"],
        "final_verdict": verdict,
        "verdict_changed_in_round3": verdict != claim1_result["verdict"],
    })

    print(json.dumps({
        "verdict_after_round2": claim1_result["verdict"],
        "final_verdict": verdict,
        "measured": result_payload,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
