#!/usr/bin/env python3
"""
Harness-initiated SCRUTINY round for Claim 1 of the Zlw4Kl5HEF reproduction.

HONESTY NOTE: this script is harness only, same rules as driver_openweights.py.
It does not decide anything about the science. A human/harness reviewer noticed
a specific technical concern with the noise model in experiments/exp_claim1.py
(the oracle's fixed variance means its correctness probability decays toward
0.5 as the query nears the root, unlike the paper's fixed-p oracle) and is
handing that concern BACK to qwen3:30b-a3b as a challenge. The MODEL decides
whether the concern is valid, whether to revise the script, and what the final
verdict is. Every prompt/response is appended to the SAME orx-openweights-trace
.jsonl the original driver wrote (this script never truncates it).

Run: ~/.venvs/icml26/bin/python challenge_claim1.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent

# Import driver_openweights.py as a module WITHOUT running its __main__ block,
# so we reuse its exact call_model / run_script / check_run / extract_code /
# extract_json / log_event machinery (and its append-only trace file).
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
    assert claim1_result["claim"].startswith("The probabilistic bisection algorithm (PBA) converges")

    original_script = SCRIPT_PATH.read_text()

    dow.log_event({
        "type": "harness_note",
        "claim": CLAIM_IDX,
        "note": (
            "SCRUTINY ROUND (post-hoc, initiated by harness/reviewer per PRIORITY-1 "
            "instructions before any publication). Rationale: a falsification from a "
            "30B local model targeting its OWN written script is at least as likely to "
            "be an implementation bug as a real finding. exp_claim1.py adds a FIXED-"
            "VARIANCE Gaussian to the continuous residual (x - x0) and thresholds the "
            "sign, which means the oracle's correctness probability Phi(|x-x0|/sigma) "
            "decays toward 0.5 as the interval shrinks near the root -- a decaying-SNR "
            "oracle, NOT the paper's classical fixed-probability-p oracle (Horstein 1963; "
            "Waeber/Frazier/Henderson 2013 formalize the exact geometric-rate theorem this "
            "paper extends). exp_claim2.py, by the same model, already implements the "
            "correct fixed-p Bernoulli oracle (P(Y=1|positive)=0.9, P(Y=1|negative)=0.1, "
            "independent of x). Confirmed against the OpenReview abstract fetched live "
            "(api2.openreview.net/notes/search) -- title/authors/abstract match paper.json "
            "exactly (Wang, Cheng, Xu; ICML 2026 regular). Feeding this concern back to the "
            "model per task instructions; the model decides whether to revise and what the "
            "final verdict is."
        ),
    })

    system_prompt = dow.SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    challenge_prompt = f"""I need you to review, and if warranted REVISE, your experiment for Claim 1 \
of this paper. This is a follow-up scrutiny round on work you already completed -- your \
original script and verdict are quoted below exactly as you produced them.

CLAIM 1 (verbatim): "{claim1_result['claim']}"

Here is the experiments/exp_claim1.py script you wrote and ran:

```python
{original_script}
```

Here is the verdict and finding you gave, based on that script's measured RESULT_JSON:
  verdict: {claim1_result['verdict']}
  finding: {claim1_result['finding']}
  measured: {json.dumps(claim1_result['measuredMetrics'])}

A careful reviewer has raised a specific technical concern about this script BEFORE this \
falsification is reported publicly. A falsification claim from a 30B local model, about its \
own script, is at least as likely to reflect an implementation bug as a real finding about the \
paper -- publishing a false falsification would be a serious credibility problem, so it must \
survive scrutiny first.

THE CONCERN: The classical Probabilistic Bisection Algorithm (Horstein 1963; the exact \
geometric-rate theorem is formalized by Waeber, Frazier & Henderson 2013, and this paper builds \
on that line) assumes a noisy oracle with a FIXED, known probability p in (0.5, 1] of returning \
the correct left/right response, INDEPENDENT of how close the query point is to the true root. \
That fixed-p oracle is the standard model this paper's theorem is about.

Your exp_claim1.py script instead adds i.i.d. Gaussian noise of FIXED VARIANCE sigma^2 directly \
to the continuous residual (x - x0), then thresholds the sign. Because the residual (x - x0) \
shrinks toward 0 as the bisection interval narrows -- which it must, since that is the whole \
point of convergence -- while sigma stays fixed, the probability of a CORRECT sign response is \
Phi(|x - x0| / sigma), which itself decays toward 0.5 as the query approaches the root. In other \
words, your oracle's reliability degrades exactly as the algorithm gets close to the answer. \
This is a fundamentally different (and provably harder) noise model than the paper's fixed-p \
oracle -- it is a known fact in this literature that a decaying-SNR / vanishing-margin oracle \
does NOT generally admit geometric convergence (this is exactly why later work introduces \
power-one sequential tests for that harder setting). So a "falsified" verdict under this noise \
model may not actually test the paper's claim; it could simply reflect that you built a \
different, harder problem than the one the paper analyzes.

For comparison: your OWN experiments/exp_claim2.py script, written for the paper's OTHER claim, \
already uses the correct construction -- a Bernoulli oracle with FIXED probabilities \
(P(Y=1|positive)=0.9, P(Y=1|negative)=0.1) that do NOT depend on distance from the true root. \
That is the paper's actual noise model, and you already know how to build it.

Please:
1. Decide for yourself whether this concern is valid -- do not just defer to the reviewer.
2. If you agree it is valid, rewrite experiments/exp_claim1.py to test Claim 1 under the \
paper's actual FIXED-PROBABILITY oracle (matching the style of your own exp_claim2.py oracle: \
a Bernoulli response correct with fixed probability p, independent of the query's distance from \
the root), at a similarly large scale (or larger) as before. Keep a genuine positive control \
that relaxes the claim's precondition -- e.g. p close to 0.5 should show markedly worse or no \
geometric convergence compared to p bounded away from 0.5 -- and also keep a direct comparison \
to noiseless binary search (p=1) as the paper's claim requires.
3. If you disagree the concern is valid, explain concretely and specifically why your original \
construction faithfully represents the paper's fixed-p oracle claim, and keep the script \
unchanged.
4. Either way, after your reasoning, give the COMPLETE current script (revised or unchanged) in \
a single fenced ```python code block, self-contained and runnable as-is, so it can be executed \
and checked for determinism exactly as before.
"""
    messages.append({"role": "user", "content": challenge_prompt})

    ok = False
    result_payload = None
    stdout_full = None
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        thinking, content = dow.call_model(messages, purpose=f"claim1_challenge_attempt{attempt}")
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
                            "note": "challenge: no code block extracted"})
            continue

        SCRIPT_PATH.write_text(code)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "wrote_challenge_revision", "path": str(SCRIPT_PATH),
                        "bytes": len(code)})

        try:
            proc1 = dow.run_script(SCRIPT_PATH)
        except Exception as e:  # subprocess.TimeoutExpired etc.
            last_error = f"Run failed/timed out: {e}"
            messages.append({"role": "user", "content": (
                f"Your script did not finish within the time budget ({e}). Please reduce "
                "scale while still faithfully targeting the claim, and reply with the FULL "
                "corrected script in a single ```python block."
            )})
            continue

        ok1, msg1, payload1 = dow.check_run(proc1)
        dow.log_event({"type": "harness_action", "claim": CLAIM_IDX, "attempt": attempt,
                        "action": "challenge_run1_result", "ok": ok1, "message": msg1,
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
                        "action": "challenge_run2_result", "ok": ok2, "message": msg2})
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
        print(f"challenge revision for claim 1 succeeded on attempt {attempt}", file=sys.stderr)
        break

    if not ok:
        print(f"challenge round FAILED after {MAX_ATTEMPTS} attempts: {last_error}", file=sys.stderr)
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": f"challenge round gave up after {MAX_ATTEMPTS} attempts",
                        "last_error": last_error})
        return 1

    # Ask the model for its final, reconsidered verdict given the new evidence.
    messages.append({"role": "user", "content": (
        "Your (possibly revised) script ran successfully twice, deterministically. Here is "
        "exactly what it printed:\n\n--- stdout ---\n" + stdout_full + "\n--- end stdout ---\n\n"
        "Now give your FINAL, reconsidered verdict for Claim 1, taking into account the "
        "scrutiny concern above and this new evidence. Follow the hard honesty rules: "
        "verified/falsified/toy/inconclusive, and every number you write in prose must trace "
        "to the RESULT_JSON dict above, to the claim text, or to a literal constant in your "
        "script. Explicitly say in your finding whether you changed your mind from your "
        "original verdict, and why or why not. Reply with a JSON object in a single ```json "
        "block with EXACTLY these keys:\n"
        '  "verdict": one of "verified", "falsified", "toy", "inconclusive"\n'
        '  "method": what you implemented and how it targets this claim (setting, scale, '
        'seeds, control)\n'
        '  "finding": what the numbers show, in prose, referencing the measured values, and '
        'whether/why your verdict changed from the original\n'
        '  "posterFinding": <=200 chars, poster-card summary\n'
        '  "scopeNote": what about the claim you did NOT cover\n'
        '  "shortClaim": <=60 chars, page title for this claim'
    )})
    thinking, content = dow.call_model(messages, purpose="claim1_challenge_verdict")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "claim": CLAIM_IDX,
                        "note": "challenge verdict JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the six keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="claim1_challenge_verdict_retry")
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
        "originalVerdictBeforeChallenge": claim1_result["verdict"],
        "originalMeasuredMetricsBeforeChallenge": claim1_result["measuredMetrics"],
    }

    session["claim_results"][0] = new_result
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))

    dow.log_event({
        "type": "challenge_end",
        "claim": CLAIM_IDX,
        "original_verdict": claim1_result["verdict"],
        "final_verdict": verdict,
        "verdict_changed": verdict != claim1_result["verdict"],
    })

    print(json.dumps({
        "original_verdict": claim1_result["verdict"],
        "final_verdict": verdict,
        "measured": result_payload,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
