#!/usr/bin/env python3
"""
Harness-initiated SCRUTINY round 2 for Claim 2 of Zlw4Kl5HEF.

Round 1 replaced the hard-cut script with a genuine Bayesian PBA (posterior
belief density on a grid, multiplicative likelihood update, query at the
posterior median) and produced verdict=falsified from avg_ratio_high=1.37,
sign_changes_high=3 (p=0.9) vs avg_ratio_low=1.28, sign_changes_low=69
(p=0.51).

The harness independently re-ran round 1's exact update logic (unmodified
algorithm, just instrumented) to sanity-check that falsification before
trusting it, and found a concrete numerical problem, reproducible and
deterministic:

  - With x_true=0.5001 and N=1000 grid points, the posterior median (the
    query point) moves for the first ~15-20 steps, then becomes PERMANENTLY
    FIXED at grid index 514 (x_t=0.514515, error=0.014415) for the remaining
    ~980 of 1000 steps -- it never changes again, regardless of further
    oracle responses. Root cause: x_true=0.5001 is only about one grid cell
    away from the algorithm's own first query (~0.4995, since the grid median
    of a uniform prior over 1000 points starts there) -- this is the same bug
    CLASS the model itself already found and fixed in Claim 1 round 4 ("root
    placed too close to the algorithm's first query point"). Once the query
    point stalls, the multiplicative likelihood update can only ever
    discriminate points on OPPOSITE sides of the (frozen) query point, never
    points on the SAME side -- so once ~70% of the density mass has piled
    onto one grid cell that happens to be 14 cells away from x_true, no
    further evidence can correct it within the remaining budget. This closely
    parallels the discretization-floor bug the model already found and fixed
    in Claim 1 round 3.
  - Same algorithm, x_true=0.37 instead (further from the initial query),
    same N=1000 grid: converges cleanly to error=0.00037 by t=999, no
    stalling.
  - Same algorithm, N=100000 instead of N=1000 (both x_true=0.5001 and
    x_true=0.37): converges cleanly to error~5e-6 in both cases, no stalling.

So the reported "falsified" verdict (ratio 1.37 >> 1, "diverging") looks like
it is measuring a root-placement + grid-resolution artifact, not a property
of the paper's actual (continuous-density) PBA. Separately, the model's
`avg_ratio` metric here is the exact "arithmetic mean of raw per-step error
ratios" statistic that the model's OWN Claim 1 finding (round 5) described as
"high-variance and upward-biased whenever the point estimate legitimately
oscillates around the truth while its envelope shrinks" -- and Claim 2 is
specifically ABOUT that oscillate-while-converging dynamic, so this is worth
the model reconsidering explicitly. There is also no near-noiseless (p->1)
positive control confirming the implementation is correct in the easy case
(Claim 1's script always included one), and only a single seed (0) is used
for each condition, which cannot yet meet AGENT_BRIEF's "show the deviation
is outside noise: multiple seeds" burden of proof for a falsification.

This round hands the model all of the above (with the harness's own
independently-reproduced numbers, so the model can verify them rather than
take the harness's word for it) and asks it to decide what to fix and
produce a final script + verdict. The harness decides nothing.

Run: /Users/tom/.venvs/icml26/bin/python challenge_claim2_round2.py
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
    round1_script = SCRIPT_PATH.read_text()

    harness_note = (
        "SCRUTINY ROUND 2 for Claim 2 (harness/reviewer-initiated, independent numerical "
        "verification of round 1's genuinely-Bayesian script, BEFORE trusting its falsified "
        "verdict). The harness re-ran round 1's exact update logic (unmodified) and found the "
        "posterior median becomes permanently stuck at a wrong grid index (514, error=0.014415) "
        "after ~15-20 of 1000 steps for x_true=0.5001 on a N=1000 grid, and never moves again -- "
        "root cause is x_true=0.5001 sitting almost exactly on the algorithm's own first query "
        "point (same bug class as Claim 1 round 4's root-placement bug), compounded by grid "
        "coarseness (same bug class as Claim 1 round 3's discretization floor). Control "
        "reruns: x_true=0.37 on the same N=1000 grid converges cleanly (error 0.00037); N=100000 "
        "converges cleanly for both root placements (error ~5e-6). Also flagging: round 1's "
        "avg_ratio metric is the same raw-ratio-mean statistic the model's own Claim 1 finding "
        "called high-variance/upward-biased for oscillating processes; only a single seed is "
        "used per condition; there is no near-noiseless positive control. Handing all of this to "
        "the model with the harness's own reproduced numbers so it can verify independently; the "
        "model decides what (if anything) to fix and what the final verdict is."
    )
    dow.log_event({"type": "harness_note", "claim": CLAIM_IDX, "note": harness_note})

    system_prompt = dow.SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    challenge_prompt = f"""This is a SECOND round of scrutiny on Claim 2, after round 1 replaced the old \
hard-cut script with a genuine Bayesian PBA. This round is about whether round 1's specific numbers can \
be trusted, not about whether the algorithm class is right (it is -- posterior density, multiplicative \
update, median query is the correct construction).

CLAIM 2 (verbatim): "{claim2_text}"

Here is your current experiments/exp_claim2.py (round 1's genuinely-Bayesian implementation):

```python
{round1_script}
```

Your round-1 verdict and finding:
  verdict: {claim2_result['verdict']}
  finding: {claim2_result['finding']}
  measured: {json.dumps(claim2_result['measuredMetrics'])}

THE ISSUE, independently verified by the harness by re-running your round-1 update logic UNMODIFIED and \
instrumenting it (you should verify this yourself rather than take the harness's word for it):

With x_true=0.5001 and N=1000 grid points, your posterior median (the query point) moves for the first \
~15-20 of 1000 steps, then becomes PERMANENTLY FIXED at grid index 514 (x_t=0.514515, error=0.014415) \
for the remaining ~980 steps -- it never changes again regardless of further oracle responses. Root \
cause: x_true=0.5001 is only about ONE grid cell away from the algorithm's own first query (the median \
of a uniform prior over 1000 points starts at ~0.4995) -- this is the SAME BUG CLASS you already found \
and fixed for Claim 1 in round 4 ("root placed too close to the algorithm's first query point"). Once \
the query point stalls at one grid index, your multiplicative update can only ever discriminate points \
on OPPOSITE sides of that (now frozen) index -- never points on the SAME side -- so once ~70% of the \
density mass has piled onto one grid cell 14 cells away from x_true, nothing in the remaining budget can \
correct it. This also parallels the discretization-floor bug you fixed in Claim 1 round 3.

Harness's control reruns of your exact algorithm (unmodified), for comparison:
  - x_true=0.37 instead of 0.5001, same N=1000 grid: converges cleanly, error=0.00037 by t=999, no stall.
  - N=100000 instead of N=1000, x_true=0.5001: converges cleanly, error~5e-6, no stall.
  - N=100000, x_true=0.37: converges cleanly, error~4e-6, no stall.

So the reported avg_ratio_high=1.37 ("diverging") looks like it measures a root-placement + \
grid-resolution artifact of THIS run's specific settings, not a property of the paper's actual PBA.

SEPARATELY, two more things worth your own reconsideration (your call on both):
1. Your `avg_ratio` metric here is the exact "arithmetic mean of raw per-step error ratios" statistic \
that YOUR OWN Claim 1 finding (round 5) described as "high-variance and upward-biased whenever the point \
estimate legitimately oscillates around the truth while its envelope shrinks" -- and Claim 2 is \
specifically ABOUT that oscillate-while-converging dynamic, so a metric known to be biased on exactly \
that dynamic deserves scrutiny here too.
2. Your current script uses a single seed (0, shared sequentially across the two run_pba calls) and no \
near-noiseless (p close to 1) positive control confirming your implementation converges correctly in the \
easy case -- both of which your Claim-1 script eventually included, and which AGENT_BRIEF requires \
before a falsification can be trusted ("show the deviation is outside noise: multiple seeds").

Please:
1. Decide for yourself whether to fix the root placement, increase the grid resolution, change the \
representation, add a positive control, add seeds, reconsider the metric, or some combination -- your \
call, and explain your reasoning.
2. If you disagree with any of the harness's diagnosis above, say so concretely and explain why, and you \
may leave the script unchanged.
3. Either way, give the COMPLETE current script (revised or unchanged) in a single fenced ```python code \
block, self-contained, deterministic (seed any seed-generation too), and runnable well within the \
5-minute budget.
"""
    messages.append({"role": "user", "content": challenge_prompt})

    ok = False
    result_payload = None
    stdout_full = None
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        thinking, content = dow.call_model(messages, purpose=f"claim2_challenge2_attempt{attempt}")
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
                            "note": "challenge2: no code block extracted"})
            continue

        SCRIPT_PATH.write_text(code)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "wrote_challenge2_revision", "path": str(SCRIPT_PATH),
                        "bytes": len(code)})

        try:
            proc1 = dow.run_script(SCRIPT_PATH)
        except Exception as e:
            last_error = f"Run failed/timed out: {e}"
            dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                            "action": "run1_timeout_or_error", "error": str(e)})
            messages.append({"role": "user", "content": (
                f"Your script did not finish within the time budget or errored ({e}). Please "
                "reduce scale if needed while keeping the genuine Bayesian update, and reply "
                "with the FULL corrected script in a single ```python block."
            )})
            continue

        ok1, msg1, payload1 = dow.check_run(proc1)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "challenge2_run1_result", "ok": ok1, "message": msg1,
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
                        "action": "challenge2_run2_result", "ok": ok2, "message": msg2})
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
        print(f"challenge round 2 for claim 2 succeeded on attempt {attempt}", file=sys.stderr)
        break

    if not ok:
        print(f"challenge round 2 for claim 2 FAILED after {MAX_ATTEMPTS} attempts: {last_error}",
              file=sys.stderr)
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": f"challenge round 2 gave up after {MAX_ATTEMPTS} attempts",
                        "last_error": last_error})
        return 1

    messages.append({"role": "user", "content": (
        "Your (possibly revised) script ran successfully twice, deterministically. Here is exactly "
        "what it printed:\n\n--- stdout ---\n" + stdout_full + "\n--- end stdout ---\n\n"
        "Give your FINAL, reconsidered verdict for Claim 2, taking into account both rounds "
        "(round 1's switch to a genuine Bayesian update, and round 2's stall/metric/control "
        "scrutiny). Follow the hard honesty rules: verified/falsified/toy/inconclusive, and every "
        "number you write in prose must trace to the RESULT_JSON dict above, to the claim text, or "
        "to a literal constant in your script. If your positive control (if you added one) does not "
        "show clean convergence, or the evidence is still mixed, the honest answer is toy or "
        "inconclusive -- that is a completely fine final answer. Summarize in your finding the full "
        "evolution across both rounds (hard-cut -> Bayesian rewrite -> stall/metric fix) and why "
        "you are confident in this final answer. Reply with a JSON object in a single ```json block "
        "with EXACTLY these keys:\n"
        '  "verdict": one of "verified", "falsified", "toy", "inconclusive"\n'
        '  "method": what you implemented and how it targets this claim (representation, scale, '
        'seeds, control)\n'
        '  "finding": what the numbers show, in prose, referencing the measured values, and the '
        'evolution across both rounds\n'
        '  "posterFinding": <=200 chars, poster-card summary\n'
        '  "scopeNote": what about the claim you did NOT cover\n'
        '  "shortClaim": <=60 chars, page title for this claim'
    )})
    thinking, content = dow.call_model(messages, purpose="claim2_challenge2_verdict")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": "challenge2 verdict JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the six keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="claim2_challenge2_verdict_retry")
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
        "verdictAfterRound1": claim2_result["verdict"],
        "measuredMetricsAfterRound1": claim2_result["measuredMetrics"],
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
        "type": "challenge_claim2_round2_end",
        "claim": CLAIM_IDX,
        "verdict_after_round1": claim2_result["verdict"],
        "final_verdict": verdict,
        "verdict_changed": verdict != claim2_result["verdict"],
    })

    print(json.dumps({
        "verdict_after_round1": claim2_result["verdict"],
        "final_verdict": verdict,
        "measured": result_payload,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
