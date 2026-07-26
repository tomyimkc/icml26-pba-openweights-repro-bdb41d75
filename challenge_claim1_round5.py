#!/usr/bin/env python3
"""
Harness-initiated SCRUTINY round 5 (TRULY FINAL) for Claim 1 of Zlw4Kl5HEF.

Rounds 1-4 fixed the oracle model, the update rule, and the discretization
floor. Round 4's "falsified" verdict rested on the WRONG SUMMARY STATISTIC:
the arithmetic mean of raw per-step error ratios errors[t+1]/errors[t]. That
statistic is high-variance and upward-biased whenever the point estimate
legitimately OSCILLATES around the truth while its envelope shrinks -- which
is exactly what the paper's OTHER official claim says PBA does ("queries
oscillate around the truth but steadily draw closer"). The harness verified
this independently (numbers below) by re-running round 4's exact update logic
with a log-linear decay slope instead of a raw-ratio mean:

  - p=1.0 (noiseless positive control): raw-ratio mean = 1.537 (looked like
    divergence) but the log-slope of error vs. step = -0.690, i.e. an implied
    per-step ratio of exp(-0.690) = 0.502 -- textbook classical-binary-search
    geometric convergence. The algorithm is almost certainly implemented
    correctly; the METRIC was wrong.
  - p=0.9 (main case): log-slope implied ratio = 0.994 -- essentially flat,
    not geometric convergence, over T=1000 steps, single seed (0).
  - p=0.51 (control): log-slope implied ratio = 1.0006 -- flat/degrading.

So with the corrected metric, the positive control now looks clean, and the
p=0.9 case still shows no visible decay -- but only on ONE seed, and the
paper's own honesty rules (and AGENT_BRIEF) require a falsification to "show
the deviation is outside noise: multiple seeds, and report the spread" before
it counts as evidence rather than an unlucky trajectory. This round hands the
model: (a) the metric fix, backed by the harness's own reproduction of the
numbers above so the model can verify them independently rather than take the
harness's word for it, and (b) the explicit multi-seed requirement. The model
decides whether to adopt the fix and what the final verdict is. No further
rounds are planned after this one.

Run: ~/.venvs/icml26/bin/python challenge_claim1_round5.py
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

    round4_script = SCRIPT_PATH.read_text()

    dow.log_event({
        "type": "harness_note",
        "claim": CLAIM_IDX,
        "note": (
            "SCRUTINY ROUND 5 -- TRULY FINAL (post-hoc, harness/reviewer-initiated). "
            "Root cause of round 4's 'falsified' verdict identified: the metric "
            "(arithmetic mean of raw per-step error ratios) is the wrong statistic "
            "for a process that legitimately oscillates while its envelope shrinks "
            "(which is exactly what the paper's OWN claim 2 describes). The harness "
            "independently re-ran round 4's exact update logic (not the model's "
            "verdict-reasoning, the actual arithmetic) with a log-linear decay-slope "
            "metric instead: p=1.0 -> implied ratio 0.502 (clean, textbook "
            "convergence -- the algorithm itself is very likely correct), p=0.9 -> "
            "0.994 (flat), p=0.51 -> 1.0006 (flat/degrading), all on a SINGLE seed "
            "(0). Per AGENT_BRIEF's own falsification burden of proof ('show the "
            "deviation is outside noise: multiple seeds, and report the spread'), a "
            "single-seed result cannot yet support a falsification claim regardless "
            "of the metric fix. Handing the model both the metric diagnosis (with "
            "the harness's own numbers so it can verify independently) and the "
            "multi-seed requirement as the FINAL round; the model decides the fix "
            "and the final verdict."
        ),
    })

    system_prompt = dow.SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    challenge_prompt = f"""This is a FIFTH and TRULY FINAL round of scrutiny on Claim 1. Rounds 1-4 fixed \
real issues (oracle model, hard-cut vs Bayesian update, discretization floor, and a root placed too \
close to the algorithm's first query). This round is about the SUMMARY STATISTIC, not the algorithm.

CLAIM 1 (verbatim): "{claim1_result['claim']}"

Here is your current experiments/exp_claim1.py (after round 4):

```python
{round4_script}
```

Your current verdict and finding:
  verdict: {claim1_result['verdict']}
  finding: {claim1_result['finding']}
  measured: {json.dumps(claim1_result['measuredMetrics'])}

THE ISSUE: your metric is `avg_ratio = mean(errors[t+1] / errors[t])`, an ARITHMETIC MEAN of raw \
per-step ratios. This statistic is high-variance and upward-biased whenever the point estimate \
legitimately OSCILLATES around the truth while its envelope (confidence width) shrinks -- which is \
EXACTLY what the paper's OTHER official claim describes: "PBA queries oscillate around the truth but \
steadily draw closer." A single step where the point estimate happens to land further from x0 than \
the previous step produces a large ratio (sometimes >>1) even while the underlying process is decaying \
geometrically in a well-defined sense (e.g. the envelope width, or the log-error trend). Averaging raw \
ratios lets a handful of such steps dominate the mean and can make a genuinely convergent process look \
divergent.

The harness independently re-ran your round-4 update logic (the exact same algorithm, unmodified) and \
computed a LOG-LINEAR DECAY SLOPE instead (fit log(error_t) vs. t by least squares over the steps before \
error hits the float64 noise floor, then report exp(slope) as the implied per-step ratio -- this is the \
standard way to characterize a claimed geometric/exponential rate, and it is insensitive to individual \
oscillation jumps): for x0=0.37, T=1000, seed=0:
  - p=1.0 (noiseless): implied ratio = exp(-0.690) = 0.502 -- clean, textbook classical-binary-search \
convergence. This strongly suggests your ALGORITHM is correct; the arithmetic-mean-of-raw-ratios metric \
was simply the wrong statistic and was hiding that.
  - p=0.9 (your main case): implied ratio = 0.994 -- essentially flat, no visible decay over 1000 steps.
  - p=0.51 (your control): implied ratio = 1.0006 -- flat / mildly degrading.

You do not have to trust these harness-computed numbers blindly -- you can and should verify them \
yourself by computing the same log-linear-slope statistic in your own script and checking it against \
your own RESULT_JSON output.

SEPARATELY, and independently of the metric: your results so far are all on a SINGLE seed (0). Per your \
own hard honesty rules, a falsification claim requires showing the deviation is outside noise -- \
multiple seeds, with the spread reported. A single trajectory cannot support a falsification either way.

Please:
1. Decide for yourself whether the log-linear-slope metric (or an equivalent geometric-mean-based \
statistic) is the right fix, and implement it alongside or instead of the raw-ratio mean.
2. Run MULTIPLE SEEDS (e.g. 10-20, as many as the time budget allows) for each of p=0.9 (main), a \
control p close to 0.5 (relaxed precondition), and p=1.0 (noiseless positive control), and report the \
mean AND spread (e.g. std or min/max) of your convergence-rate statistic across seeds for each p.
3. Verify your p=1.0 positive control now shows clean, strong, LOW-VARIANCE convergence across seeds \
before trusting any falsified/verified conclusion about p=0.9.
4. If you disagree with any of this, explain concretely why, and keep the script unchanged.
5. Either way, after your reasoning, give the COMPLETE current script (revised or unchanged) in a \
single fenced ```python code block, self-contained, deterministic (seed the seed generation itself too), \
and runnable as-is within the 5-minute budget.
"""
    messages.append({"role": "user", "content": challenge_prompt})

    ok = False
    result_payload = None
    stdout_full = None
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        thinking, content = dow.call_model(messages, purpose=f"claim1_challenge5_attempt{attempt}")
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
                            "note": "challenge5: no code block extracted"})
            continue

        SCRIPT_PATH.write_text(code)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "wrote_challenge5_revision", "path": str(SCRIPT_PATH),
                        "bytes": len(code)})

        try:
            proc1 = dow.run_script(SCRIPT_PATH)
        except Exception as e:
            last_error = f"Run failed/timed out: {e}"
            messages.append({"role": "user", "content": (
                f"Your script did not finish within the time budget ({e}). Please reduce "
                "the number of seeds or T if needed while keeping multiple seeds and the "
                "log-slope-style metric, and reply with the FULL corrected script in a "
                "single ```python block."
            )})
            continue

        ok1, msg1, payload1 = dow.check_run(proc1)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "challenge5_run1_result", "ok": ok1, "message": msg1,
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
                        "action": "challenge5_run2_result", "ok": ok2, "message": msg2})
        if not ok2 or payload1 != payload2:
            last_error = f"Non-deterministic or failed second run: {msg2}"
            messages.append({"role": "user", "content": (
                "Your script's two runs did not match / the second run failed:\n"
                f"  run1: {json.dumps(payload1)}\n  run2 msg: {msg2}\n"
                "This must be fully deterministic (seed the seed-generation too). Please "
                "fix and reply with the FULL corrected script in a single ```python block."
            )})
            continue

        ok = True
        result_payload = payload1
        stdout_full = proc1.stdout
        print(f"challenge round 5 revision for claim 1 succeeded on attempt {attempt}", file=sys.stderr)
        break

    if not ok:
        print(f"challenge round 5 FAILED after {MAX_ATTEMPTS} attempts: {last_error}", file=sys.stderr)
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": f"challenge round 5 gave up after {MAX_ATTEMPTS} attempts",
                        "last_error": last_error})
        return 1

    messages.append({"role": "user", "content": (
        "Your (possibly revised) script ran successfully twice, deterministically. Here is "
        "exactly what it printed:\n\n--- stdout ---\n" + stdout_full + "\n--- end stdout ---\n\n"
        "This is the TRULY FINAL round -- give your FINAL, reconsidered verdict for Claim 1, "
        "taking into account all FIVE rounds of scrutiny (oracle model, hard-cut vs Bayesian "
        "update, discretization floor, root placement, and the metric/multi-seed fix in this "
        "round). Check that your p=1.0 positive control shows clean, low-variance convergence "
        "across seeds; if it does, you can trust whatever the p=0.9 and control-p results show. "
        "If your positive control still does not look clean, the honest verdict remains "
        "inconclusive or toy -- that is a completely fine final answer. Follow the hard honesty "
        "rules: every number you write in prose must trace to the RESULT_JSON dict above, to the "
        "claim text, or to a literal constant in your script. Summarize in your finding the full "
        "evolution across all five scrutiny rounds and why you are confident in this final answer. "
        "Reply with a JSON object in a single ```json block with EXACTLY these keys:\n"
        '  "verdict": one of "verified", "falsified", "toy", "inconclusive"\n'
        '  "method": what you implemented and how it targets this claim (setting, scale, '
        'seeds, control)\n'
        '  "finding": what the numbers show, in prose, referencing the measured values, and '
        'the full evolution of your verdict across all five scrutiny rounds\n'
        '  "posterFinding": <=200 chars, poster-card summary\n'
        '  "scopeNote": what about the claim you did NOT cover\n'
        '  "shortClaim": <=60 chars, page title for this claim'
    )})
    thinking, content = dow.call_model(messages, purpose="claim1_challenge5_verdict")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": "challenge5 verdict JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the six keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="claim1_challenge5_verdict_retry")
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
        "verdictAfterRound4": claim1_result["verdict"],
        "measuredMetricsAfterRound4": claim1_result["measuredMetrics"],
    }

    session["claim_results"][0] = new_result
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))

    dow.log_event({
        "type": "challenge_round5_end",
        "claim": CLAIM_IDX,
        "verdict_after_round4": claim1_result["verdict"],
        "final_verdict": verdict,
        "verdict_changed_in_round5": verdict != claim1_result["verdict"],
    })

    print(json.dumps({
        "verdict_after_round4": claim1_result["verdict"],
        "final_verdict": verdict,
        "measured": result_payload,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
