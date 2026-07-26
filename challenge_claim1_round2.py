#!/usr/bin/env python3
"""
Harness-initiated SCRUTINY round 2 for Claim 1 of the Zlw4Kl5HEF reproduction.

Same honesty rules as challenge_claim1.py: harness only, no verdict decided here.
Round 1 fixed the oracle noise model (now a genuine fixed-p Bernoulli oracle,
independent of query-to-root distance) but the verdict stayed "falsified" using
a HARD interval-bisection update rule. A second, more fundamental concern:
hard interval bisection is irreversible -- once one wrong oracle response
occurs (near-certain over T=100 steps for any p<1), the true root is
permanently excluded from [a,b] and no later correct response can recover it.
That is NOT what the actual Probabilistic Bisection Algorithm (Horstein 1963)
does: PBA maintains a full-support belief density/mass function over the whole
domain and updates it multiplicatively (soft evidence), so a wrong response
only shifts probability mass -- it never drives any point's belief to exactly
zero when 0<p<1, and the algorithm can and does recover. That reversibility is
the mechanistic content of the paper's claim 2 ("queries oscillate ... but
steadily draw closer"), so a hard-cut script cannot faithfully test claim 1
either. This round hands that concern to the model, together with the standard
discretized-belief PBA update rule (public literature, not implementation
detail invented by the harness), and lets the model decide whether to revise
and what the final verdict is.

Run: ~/.venvs/icml26/bin/python challenge_claim1_round2.py
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

    round1_script = SCRIPT_PATH.read_text()

    dow.log_event({
        "type": "harness_note",
        "claim": CLAIM_IDX,
        "note": (
            "SCRUTINY ROUND 2 (post-hoc, harness/reviewer-initiated). Round 1 fixed "
            "the oracle noise model (now correctly fixed-p, distance-independent) but "
            "kept verdict=falsified using HARD interval bisection: `if response==1: "
            "b=mid else: a=mid`. This is irreversible -- once ANY wrong response occurs "
            "(P(>=1 wrong in T=100 steps) = 1-p^100, e.g. ~99.997% at p=0.9), the true "
            "root is permanently excluded from [a,b] and no later correct response can "
            "recover it. The actual Horstein/PBA algorithm this paper analyzes instead "
            "keeps a full-support belief distribution and updates it multiplicatively "
            "(soft evidence), which is reversible by construction whenever 0<p<1 -- that "
            "reversibility is exactly what the paper's OTHER claim ('queries oscillate "
            "around the truth but steadily draw closer') describes. Feeding this back to "
            "the model with the standard discretized-belief PBA update rule (public "
            "literature: Horstein 1963; Waeber, Frazier & Henderson 2013 formalize the "
            "geometric-rate theorem this paper extends) and letting the model decide "
            "whether to revise and what the final verdict is."
        ),
    })

    system_prompt = dow.SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    challenge_prompt = f"""This is a SECOND round of scrutiny on Claim 1, after your first revision. Your \
first revision correctly fixed the oracle to be a fixed-probability Bernoulli response \
(independent of distance to the root) -- good. But a second, more fundamental concern \
has been raised about the ALGORITHM ITSELF, and it needs your judgment before this result \
can be reported as a falsification.

CLAIM 1 (verbatim): "{claim1_result['claim']}"

Here is your current experiments/exp_claim1.py (after round 1's fix):

```python
{round1_script}
```

Your current verdict and finding:
  verdict: {claim1_result['verdict']}
  finding: {claim1_result['finding']}
  measured: {json.dumps(claim1_result['measuredMetrics'])}

THE NEW CONCERN: Your script updates the search interval with a HARD, irreversible cut:
    if response == 1: b = mid
    else:              a = mid
Once a single WRONG oracle response occurs, the true root x0 is permanently excluded from \
[a, b]. Because the oracle is wrong with probability (1-p) at every one of T=100 independent \
steps, the probability of at least one wrong response is 1 - p^100 -- about 99.997% even at \
p=0.9. So on almost every run, the interval permanently loses the root at some point, and no \
number of subsequent CORRECT responses can bring it back -- the algorithm has no mechanism to \
detect or undo that earlier mistake. That would explain persistent non-convergence (ratio >= 1) \
for essentially any p < 1, regardless of how large p is. This is a structural property of hard \
bisection, not evidence about the paper's actual algorithm.

The real Probabilistic Bisection Algorithm (Horstein, IEEE Trans. Info. Theory 1963; the exact \
geometric-rate theorem this paper proves is the modern formalization by Waeber, Frazier & \
Henderson, SIAM J. Control Optim. 2013) does NOT hard-cut an interval. It maintains a belief \
probability distribution f_t over the WHOLE domain (e.g. a uniform prior on [0,1] at t=0) and \
updates it by BAYES' RULE (multiplicative, soft evidence) after each noisy response, so no \
candidate location's belief is ever driven to exactly zero while 0 < p < 1 -- the algorithm can \
and does recover from a wrong response over time. That reversibility is exactly what the paper's \
OTHER official claim describes: "PBA queries oscillate around the truth but steadily draw closer." \
A hard-cut interval cannot oscillate back toward a wrongly-excluded root at all, so it cannot be a \
faithful test of either claim.

The standard discretized (grid/histogram) implementation, which you can implement directly:
  1. Discretize [0, 1] into N bins (as large as your time budget allows, e.g. N=2000+), belief \
array f initialized uniform (f[i] = 1/N for all i).
  2. At each step, query x_t = the point where the CUMULATIVE belief crosses 0.5 (the median of \
the current belief distribution) -- NOT necessarily the arithmetic midpoint of any interval.
  3. Draw a noisy response: with probability p the oracle correctly reports whether the true root \
x0 is to the left or right of x_t; with probability (1-p) it reports the opposite. This p does \
NOT depend on |x_t - x0|.
  4. Bayesian update (soft, multiplicative, never a hard cut): if the response says "root is to \
the right of x_t", multiply every bin center c > x_t by p and every bin center c <= x_t by (1-p); \
if the response says "root is to the left of x_t", do the reverse. Then renormalize f so it sums \
to 1 again.
  5. The point estimate at each step is again the median of the (updated) belief distribution; \
track its distance to x0 over time, exactly as before.

Please:
1. Decide for yourself whether this concern is valid -- do not just defer to the reviewer.
2. If you agree, rewrite experiments/exp_claim1.py to implement this discretized Bayesian PBA \
(not a hard interval cut) at the largest scale (N bins, T steps) that fits the time budget. Keep \
a fixed p bounded away from 0.5 as the main condition, a genuine positive control that relaxes \
the claim's precondition (e.g. p very close to 0.5, which should show markedly worse convergence \
than p bounded away from 0.5), and the noiseless binary search comparison (p=1) the claim \
requires.
3. If you disagree the concern is valid, explain concretely and specifically why hard interval \
bisection under a fixed-p oracle still faithfully represents the paper's PBA claim despite the \
irreversibility argument above, and keep the script unchanged.
4. Either way, after your reasoning, give the COMPLETE current script (revised or unchanged) in \
a single fenced ```python code block, self-contained and runnable as-is.
"""
    messages.append({"role": "user", "content": challenge_prompt})

    ok = False
    result_payload = None
    stdout_full = None
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        thinking, content = dow.call_model(messages, purpose=f"claim1_challenge2_attempt{attempt}")
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
            messages.append({"role": "user", "content": (
                f"Your script did not finish within the time budget ({e}). Please reduce "
                "scale (smaller N/T) while still faithfully implementing the discretized "
                "Bayesian update, and reply with the FULL corrected script in a single "
                "```python block."
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
        print(f"challenge round 2 revision for claim 1 succeeded on attempt {attempt}", file=sys.stderr)
        break

    if not ok:
        print(f"challenge round 2 FAILED after {MAX_ATTEMPTS} attempts: {last_error}", file=sys.stderr)
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": f"challenge round 2 gave up after {MAX_ATTEMPTS} attempts",
                        "last_error": last_error})
        return 1

    messages.append({"role": "user", "content": (
        "Your (possibly revised) script ran successfully twice, deterministically. Here is "
        "exactly what it printed:\n\n--- stdout ---\n" + stdout_full + "\n--- end stdout ---\n\n"
        "Now give your FINAL, reconsidered verdict for Claim 1, taking into account BOTH rounds "
        "of scrutiny (the oracle noise model in round 1, and the hard-cut-vs-Bayesian-update "
        "concern in round 2) and this new evidence. Follow the hard honesty rules: "
        "verified/falsified/toy/inconclusive, and every number you write in prose must trace "
        "to the RESULT_JSON dict above, to the claim text, or to a literal constant in your "
        "script. Explicitly say in your finding whether/why your verdict changed across both "
        "rounds. Reply with a JSON object in a single ```json block with EXACTLY these keys:\n"
        '  "verdict": one of "verified", "falsified", "toy", "inconclusive"\n'
        '  "method": what you implemented and how it targets this claim (setting, scale, '
        'seeds, control)\n'
        '  "finding": what the numbers show, in prose, referencing the measured values, and '
        'whether/why your verdict changed across both scrutiny rounds\n'
        '  "posterFinding": <=200 chars, poster-card summary\n'
        '  "scopeNote": what about the claim you did NOT cover\n'
        '  "shortClaim": <=60 chars, page title for this claim'
    )})
    thinking, content = dow.call_model(messages, purpose="claim1_challenge2_verdict")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": "challenge2 verdict JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the six keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="claim1_challenge2_verdict_retry")
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
        "verdictAfterRound1": claim1_result["verdict"],
        "measuredMetricsAfterRound1": claim1_result["measuredMetrics"],
    }

    session["claim_results"][0] = new_result
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))

    dow.log_event({
        "type": "challenge_round2_end",
        "claim": CLAIM_IDX,
        "verdict_after_round1": claim1_result["verdict"],
        "final_verdict": verdict,
        "verdict_changed_in_round2": verdict != claim1_result["verdict"],
    })

    print(json.dumps({
        "verdict_after_round1": claim1_result["verdict"],
        "final_verdict": verdict,
        "measured": result_payload,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
