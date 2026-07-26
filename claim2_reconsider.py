#!/usr/bin/env python3
"""
One quick follow-up turn: after 5 rounds of scrutiny fixed real issues in
Claim 1's script (a hard interval-cut update rule was one of them, replaced
with a genuine Bayesian belief-density update), ask the model whether that
same simplification in its UNCHANGED Claim 2 script affects its confidence in
Claim 2's verdict or scope note. Harness only: presents the fact (Claim 2's
script still uses the original hard-cut update rule, single seed), asks the
model to decide whether to revise its own scopeNote/verdict, and lets the
model author whatever text results. No new experiment is forced.

Run: ~/.venvs/icml26/bin/python claim2_reconsider.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("dow", PAPER_DIR / "driver_openweights.py")
dow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dow)

SESSION_OUT_PATH = PAPER_DIR / "openweights_session_output.json"
CLAIM2_SCRIPT = PAPER_DIR / "experiments" / "exp_claim2.py"


def main() -> int:
    session = json.loads(SESSION_OUT_PATH.read_text())
    claim2_result = session["claim_results"][1]
    claim1_result = session["claim_results"][0]
    script_text = CLAIM2_SCRIPT.read_text()

    dow.log_event({"type": "harness_note", "claim": 2, "note": (
        "Post-hoc consistency check (harness-initiated). Claim 1's scrutiny rounds "
        "found that a HARD interval-cut update rule (irreversible on a single wrong "
        "response) is a different, weaker algorithm than genuine Bayesian belief-"
        "density PBA, and fixed it. Claim 2's script (unchanged since the original "
        "session) still uses the same hard-cut update rule, and its convergence "
        "margin is thin (avg_ratio_sym=0.998, barely below 1) on a single seed. "
        "Asking the model whether this affects its Claim 2 verdict or scope note; "
        "no new experiment forced, no verdict decided by the harness."
    )})

    messages = [{"role": "system", "content": dow.SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": f"""One quick consistency check before this reproduction is finalized. You just spent \
5 rounds fixing Claim 1's script, and one real bug you fixed there (round 2) was a HARD interval-cut \
update rule (`if response==1: b=mid else: a=mid`) -- irreversible on a single wrong oracle response -- \
which you replaced with a genuine Bayesian belief-density update.

Your Claim 2 script (unchanged since your original session) uses that SAME hard-cut update rule:

```python
{script_text}
```

Your current Claim 2 verdict and finding:
  verdict: {claim2_result['verdict']}
  finding: {claim2_result['finding']}
  measured: {json.dumps(claim2_result['measuredMetrics'])}

Question: given what you now know about hard-cut vs. Bayesian updates from Claim 1, does this affect \
your confidence in Claim 2's verdict or scope note? Note Claim 2's own numbers are already fairly thin \
(avg_ratio_sym=0.998, barely below 1; sign_changes_sym=1, i.e. only one oscillation over 1000 steps; \
single seed only) -- for context, Claim 1's FINAL verdict is: {claim1_result['verdict']}, with the reason \
being: {claim1_result['finding'][:400]}

You do not need to rewrite the script or verdict unless you judge it necessary -- you may also decide \
the hard-cut simplification is an acceptable approximation for THIS claim (which is about oscillation/ \
concentration dynamics, not the geometric RATE Claim 1 is about) and leave it unchanged. Either way, \
give your honest, final scopeNote and verdict for Claim 2. Reply with a JSON object in a single ```json \
block with EXACTLY these keys:
  "verdict": one of "verified", "falsified", "toy", "inconclusive"
  "scopeNote": your final, honest scope note for Claim 2 (update it if this consideration changes what \
you think the honest scope caveats are; keep it if you judge no change is warranted)
  "reasoning": 1-2 sentences on whether/why the hard-cut concern from Claim 1 does or does not apply here
"""})
    thinking, content = dow.call_model(messages, purpose="claim2_reconsider")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "claim": 2, "note": "reconsider JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the three keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="claim2_reconsider_retry")
        messages.append({"role": "assistant", "content": content})
        fields = dow.extract_json(content) or {}

    verdict = str(fields.get("verdict", claim2_result["verdict"])).lower()
    scope_note = fields.get("scopeNote") or claim2_result["scopeNote"]
    session["claim_results"][1]["verdict"] = verdict
    session["claim_results"][1]["scopeNote"] = scope_note
    session["claim_results"][1]["reconsiderReasoning"] = fields.get("reasoning", "")
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))

    dow.log_event({"type": "claim2_reconsider_end", "verdict": verdict, "scopeNote": scope_note,
                    "reasoning": fields.get("reasoning", "")})
    print(json.dumps(fields, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
