#!/usr/bin/env python3
"""
Harness-initiated SCRUTINY round 1 for Claim 2 of Zlw4Kl5HEF, driven by an
INDEPENDENT JUDGE's verbatim scoring feedback (not harness-invented critique).

Background: Claim 2's original experiments/exp_claim2.py used a hard
interval-cut update rule (binary-search style: bisect [L, R] and throw away
one half based on a single noisy oracle response). The model itself caught
that mismatch in claim2_reconsider.py / claim2_reconsider2.py and honestly
downgraded its own verdict to `inconclusive` -- the harness agrees that call
was correct FOR THAT SCRIPT.

An independent judge then scored the reproduction and gave Claim 2 zero
points (inconclusive = 0 by the challenge's own scoring table), with this
reasoning, quoted VERBATIM and unedited:

  "The Claim 2 experiment uses a hard-cut interval update rule (binary search
  style) rather than the paper's Bayesian belief-density update, as
  acknowledged in the logbook itself. The measured results
  (avg_ratio_sym=0.998, sign_changes_sym=1) are from the wrong algorithm and
  cannot establish or refute anything about PBA's oscillation-and-
  concentration dynamics."

This is not a new criticism -- it is confirmation that the earlier
self-correction was right. The gap this round closes is that nobody has yet
implemented the REAL algorithm, so Claim 2 has never actually been tested.
This round hands the model the judge's feedback verbatim, plus harness
background on what "the paper's actual Bayesian PBA" concretely refers to (a
posterior belief density over the root, updated multiplicatively by the
oracle's likelihood at each query, queried at the posterior MEDIAN -- the
standard Horstein / Waeber-Frazier-Henderson construction), and asks the
model to design and write a NEW experiments/exp_claim2.py implementing it.
The model decides the concrete representation, domain, grid/resolution (or
alternative), noise level, scale, seeds, metrics, and control -- the harness
only executes exactly what it writes and reports stdout/stderr back, exactly
as every other round of this reproduction.

Run: /Users/tom/.venvs/icml26/bin/python challenge_claim2_round1.py
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

JUDGE_FEEDBACK = (
    "The Claim 2 experiment uses a hard-cut interval update rule (binary search "
    "style) rather than the paper's Bayesian belief-density update, as "
    "acknowledged in the logbook itself. The measured results "
    "(avg_ratio_sym=0.998, sign_changes_sym=1) are from the wrong algorithm and "
    "cannot establish or refute anything about PBA's oscillation-and-"
    "concentration dynamics."
)


def main() -> int:
    session = json.loads(SESSION_OUT_PATH.read_text())
    claim2_result = session["claim_results"][1]
    claim2_text = claim2_result["claim"]
    old_script = SCRIPT_PATH.read_text()

    dow.log_event({
        "type": "harness_note",
        "claim": CLAIM_IDX,
        "note": (
            "SCRUTINY ROUND 1 for Claim 2, triggered by an INDEPENDENT JUDGE's "
            "scoring feedback (external to this harness/model session), quoted "
            "verbatim below. The judge scored the prior honest `inconclusive` "
            "verdict as 0 points and confirmed the root cause the model had "
            "already self-diagnosed: the script implements a hard interval-cut, "
            "not the paper's actual Bayesian PBA. This round hands the model the "
            "judge feedback verbatim plus background on the standard Bayesian-PBA "
            "construction (posterior belief density, multiplicative likelihood "
            "update, query at the posterior median) and asks it to design and "
            "implement the real algorithm so Claim 2 can be genuinely assessed. "
            "The model decides representation, scale, seeds, metrics, and "
            "control; the harness executes exactly what it writes."
        ),
        "judge_feedback_verbatim": JUDGE_FEEDBACK,
    })

    system_prompt = dow.SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    challenge_prompt = f"""An INDEPENDENT JUDGE reviewed this reproduction and scored Claim 2 as \
`inconclusive` (0 points). Here is the judge's reasoning, quoted VERBATIM and unedited -- respond to \
this actual criticism, not a paraphrase of it:

  "{JUDGE_FEEDBACK}"

The judge is right, and it matches what you yourself already found in claim2_reconsider2.py: the current \
experiments/exp_claim2.py implements a hard interval-cut (binary-search-style) update, not the paper's \
actual Bayesian PBA. Your `inconclusive` verdict for THAT script was the honest, correct call, and it \
still stands as the right verdict FOR A SCRIPT THAT DOESN'T IMPLEMENT THE ALGORITHM. But that also means \
Claim 2 has never actually been tested. This round's job is to fix that: implement the REAL algorithm so \
Claim 2 can be genuinely assessed -- verified, falsified, or (if you still cannot faithfully test it, or \
the evidence is mixed) toy or inconclusive again, honestly.

CLAIM 2 (verbatim): "{claim2_text}"

For reference, here is your current (hard-cut, judged-inadequate) experiments/exp_claim2.py:

```python
{old_script}
```

BACKGROUND on what "the paper's actual Bayesian PBA" refers to -- the standard construction in the PBA \
literature (Horstein 1963; formalized for stochastic root-finding by Waeber, Frazier & Henderson 2013). \
This is background only, not a spec you must copy verbatim -- you still decide the concrete \
implementation and must defend your choices:

- Maintain a posterior belief density f_t(x) over the location of the true root x*, typically \
initialized as uniform over the search domain [0, 1].
- At each step t, query the CURRENT POSTERIOR MEDIAN x_t (the point that splits the belief mass exactly \
in half) -- NOT the midpoint of a shrinking interval. This is the crux of the difference from a hard-cut \
bisection.
- A (possibly noisy) oracle returns a sign Y_t indicating which side of x_t it believes the root is on, \
correct with a known probability p > 0.5.
- UPDATE THE FULL DENSITY multiplicatively by the oracle's likelihood everywhere, not by cutting the \
support in two:
    f_{{t+1}}(x) is proportional to f_t(x) * L(Y_t | x)
  where L(Y_t | x) = p if the sign of (x* - x) implied by comparing x to x_t agrees with Y_t, else (1-p). \
  Points "on the wrong side" of a single noisy answer keep nonzero mass under this update (they are only \
  reweighted down, never discarded) -- which is exactly the mechanism that would let a later query \
  legitimately land back on the other side of the true root (oscillation) even while the belief mass as a \
  whole keeps concentrating over time. That is the dynamic Claim 2 describes.
- One convenient, exactly-representable way to implement this deterministically is a piecewise-constant \
density on a fine grid over the domain, renormalized after each multiplicative update -- but that is only \
one valid choice; use whatever representation you can make exact, fast, and deterministic.

You decide: the domain and grid resolution (or an alternative representation entirely), the oracle noise \
level p, the number of steps T, the number of seeds, and -- per the AGENT_BRIEF contract -- a control that \
relaxes Claim 2's precondition (for example: an oracle so noisy it is barely informative, or one that is \
adversarially biased) and shows the oscillation/concentration property degrades. Decide for yourself what \
to actually measure to characterize "queries oscillate around the truth but steadily draw closer, \
yielding an estimator that rapidly concentrates on the truth" -- for instance (not prescriptive) the sign \
of (x_t - x*) over time to characterize oscillation, and some measure of how fast the posterior mass \
concentrates near x* over time (e.g. the width of the interval containing a fixed fraction of the mass, \
or the posterior variance) to characterize concentration.

Follow the AGENT_BRIEF contract exactly: Python stdlib / numpy / scipy / pandas / matplotlib only, no \
network, no GPU, no torch; fully deterministic (seed every RNG with np.random.default_rng, including any \
seed-generation step); runs well under 5 minutes; and prints EXACTLY one `RESULT_JSON` line containing \
every number you will later want to discuss in prose.

First, in 1-3 short paragraphs, explain your design: how you represent and update the posterior, at what \
scale, exactly what you measure and why it characterizes oscillation-and-concentration, and what control \
you include. Then give the COMPLETE, self-contained script in a single fenced ```python code block.
"""
    messages.append({"role": "user", "content": challenge_prompt})

    ok = False
    result_payload = None
    stdout_full = None
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        thinking, content = dow.call_model(messages, purpose=f"claim2_challenge1_attempt{attempt}")
        messages.append({"role": "assistant", "content": content})
        code = dow.extract_code(content)
        if not code:
            last_error = "No fenced python code block found in response."
            messages.append({"role": "user", "content": (
                "I could not find a fenced ```python code block in your last reply. Please "
                "reply again with the FULL script (revised) in a single ```python block."
            )})
            dow.log_event({"type": "harness_note", "claim": CLAIM_IDX, "attempt": attempt,
                            "note": "challenge1: no code block extracted"})
            continue

        SCRIPT_PATH.write_text(code)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "wrote_claim2_bayesian_revision", "path": str(SCRIPT_PATH),
                        "bytes": len(code)})

        try:
            proc1 = dow.run_script(SCRIPT_PATH)
        except Exception as e:
            last_error = f"Run failed/timed out: {e}"
            dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                            "action": "run1_timeout_or_error", "error": str(e)})
            messages.append({"role": "user", "content": (
                f"Your script did not finish within the time budget or errored ({e}). Please "
                "reduce the grid resolution / T / number of seeds if needed while keeping a "
                "genuine Bayesian belief-density update and a control, and reply with the FULL "
                "corrected script in a single ```python block."
            )})
            continue

        ok1, msg1, payload1 = dow.check_run(proc1)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "challenge1_run1_result", "ok": ok1, "message": msg1,
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
                        "action": "challenge1_run2_result", "ok": ok2, "message": msg2})
        if not ok2 or payload1 != payload2:
            last_error = f"Non-deterministic or failed second run: {msg2}"
            dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                            "action": "determinism_mismatch", "run1": payload1, "run2_msg": msg2})
            messages.append({"role": "user", "content": (
                "Your script's two runs did not match / the second run failed:\n"
                f"  run1: {json.dumps(payload1)}\n  run2 msg: {msg2}\n"
                "This must be fully deterministic (seed the seed-generation too, and avoid any "
                "dict/set iteration-order or wall-clock dependence). Please fix and reply with "
                "the FULL corrected script in a single ```python block."
            )})
            continue

        ok = True
        result_payload = payload1
        stdout_full = proc1.stdout
        print(f"challenge round 1 (Bayesian rewrite) for claim 2 succeeded on attempt {attempt}",
              file=sys.stderr)
        break

    if not ok:
        print(f"challenge round 1 for claim 2 FAILED after {MAX_ATTEMPTS} attempts: {last_error}",
              file=sys.stderr)
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": f"challenge round 1 gave up after {MAX_ATTEMPTS} attempts",
                        "last_error": last_error})
        return 1

    messages.append({"role": "user", "content": (
        "Your (revised) script ran successfully twice, deterministically. Here is exactly what "
        "it printed:\n\n--- stdout ---\n" + stdout_full + "\n--- end stdout ---\n\n"
        "Now give your verdict for Claim 2, based ONLY on this new, genuinely-Bayesian "
        "implementation (not the old hard-cut script, which is now superseded). Follow the hard "
        "honesty rules: verified/falsified/toy/inconclusive, and every number you write in prose "
        "must trace to the RESULT_JSON dict above, to the claim text, or to a literal constant in "
        "your script. Consider your control explicitly. If the new implementation still cannot "
        "cleanly resolve the claim (e.g. the control does not degrade as expected, or the "
        "evidence is genuinely mixed), the honest answer is toy or inconclusive -- that is a "
        "completely fine final answer; do not inflate it. Reply with a JSON object in a single "
        "```json block with EXACTLY these keys:\n"
        '  "verdict": one of "verified", "falsified", "toy", "inconclusive"\n'
        '  "method": what you implemented and how it targets this claim (representation, scale, '
        'seeds, control) and how it differs from the old, judge-rejected hard-cut script\n'
        '  "finding": what the numbers show, in prose, referencing the measured values\n'
        '  "posterFinding": <=200 chars, poster-card summary\n'
        '  "scopeNote": what about the claim you did NOT cover\n'
        '  "shortClaim": <=60 chars, page title for this claim'
    )})
    thinking, content = dow.call_model(messages, purpose="claim2_challenge1_verdict")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": "challenge1 verdict JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the six keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="claim2_challenge1_verdict_retry")
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
        "verdictBeforeBayesianRound1": claim2_result["verdict"],
        "measuredMetricsBeforeBayesianRound1": claim2_result["measuredMetrics"],
        "judgeFeedbackVerbatim": JUDGE_FEEDBACK,
    }

    session["claim_results"][1] = new_result
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))

    # Also update the results.json spec that the builder actually reads.
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
        "type": "challenge_claim2_round1_end",
        "claim": CLAIM_IDX,
        "verdict_before": claim2_result["verdict"],
        "final_verdict": verdict,
        "verdict_changed": verdict != claim2_result["verdict"],
    })

    print(json.dumps({
        "verdict_before": claim2_result["verdict"],
        "final_verdict": verdict,
        "measured": result_payload,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
