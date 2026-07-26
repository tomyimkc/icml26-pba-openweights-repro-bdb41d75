#!/usr/bin/env python3
"""
Follow-up to claim2_reconsider.py: the model's answer was internally
inconsistent -- it said the hard-cut script "does not match the paper's
algorithm" and is "irrelevant to the paper's claim" / "invalidating the
test", but then assigned verdict=falsified rather than inconclusive. Per the
AGENT_BRIEF honesty rules, a test that does not faithfully target the claim
cannot establish a falsification (or a verification) -- that is precisely
what "inconclusive" means. This turn points out the inconsistency directly
and asks the model to reconcile it. Harness only: no verdict decided here.

Run: ~/.venvs/icml26/bin/python claim2_reconsider2.py
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


def main() -> int:
    session = json.loads(SESSION_OUT_PATH.read_text())
    claim2_result = session["claim_results"][1]

    dow.log_event({"type": "harness_note", "claim": 2, "note": (
        "Internal-consistency check on the previous turn's answer: the model "
        "said the hard-cut script 'does not match the paper's algorithm' and is "
        "'irrelevant to the paper's claim' / 'invalidating the test', yet assigned "
        "verdict=falsified rather than inconclusive. Per the honesty rules, a test "
        "that does not faithfully target the claim cannot establish a falsification "
        "(or a verification) -- that is what 'inconclusive' is for. Pointing this "
        "out directly and asking the model to reconcile; harness decides nothing."
    )})

    messages = [{"role": "system", "content": dow.SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": f"""Your last answer on Claim 2 was internally inconsistent, and needs a direct reconciliation \
before this reproduction is finalized.

You said: "{claim2_result.get('reconsiderReasoning', '')}"

And your scope note said: "{claim2_result['scopeNote']}"

But you assigned verdict = "{claim2_result['verdict']}".

Here is the inconsistency: if the hard-cut update rule "does not match the paper's algorithm" and is \
"irrelevant to the paper's claim" -- i.e. the script does not faithfully test what Claim 2 asserts -- \
then the measured numbers cannot establish a FALSIFICATION either. A test that does not faithfully \
target the claim is not evidence the claim is false; it is simply not evidence about the claim at all. \
Per your own hard honesty rules: "If you genuinely cannot target a claim, mark it inconclusive and say \
why. That is an honest 0, and it is much better than a fake verified" -- the same logic means it would \
also be a mistake to report a fake falsified when the implementation does not faithfully test the claim.

Please resolve this directly. You have two honest options, and you decide which is correct:
(a) You believe, on reflection, that the hard-cut update rule is NOT close enough to genuine Bayesian \
PBA to say anything about Claim 2 (an oscillation/concentration claim, not a rate claim) -- in which \
case the honest verdict is "inconclusive", with a scope note explaining that the implementation used \
does not match the paper's actual algorithm closely enough to draw a conclusion either way.
(b) You believe, on reflection, that despite being a simplification, the hard-cut version still \
reasonably exercises the oscillation/concentration mechanism Claim 2 describes (bounded interval size, \
noisy responses, tracking whether the query alternates sides while narrowing) -- in which case your \
ORIGINAL verdict and finding (before this consistency check) may still be the honest answer, and you \
should say so plainly, with a scope note that honestly notes the simplification without overclaiming.

Give your FINAL answer for Claim 2. Reply with a JSON object in a single ```json block with EXACTLY \
these keys:
  "verdict": one of "verified", "falsified", "toy", "inconclusive"
  "scopeNote": your final, honest, internally-consistent scope note for Claim 2
  "reasoning": 1-3 sentences explaining which of (a) or (b) you chose and why, addressing the \
inconsistency directly
"""})
    thinking, content = dow.call_model(messages, purpose="claim2_reconsider2")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "claim": 2, "note": "reconsider2 JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the three keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="claim2_reconsider2_retry")
        messages.append({"role": "assistant", "content": content})
        fields = dow.extract_json(content) or {}

    verdict = str(fields.get("verdict", claim2_result["verdict"])).lower()
    scope_note = fields.get("scopeNote") or claim2_result["scopeNote"]
    session["claim_results"][1]["verdict"] = verdict
    session["claim_results"][1]["scopeNote"] = scope_note
    session["claim_results"][1]["reconsiderReasoning2"] = fields.get("reasoning", "")
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))

    dow.log_event({"type": "claim2_reconsider2_end", "verdict": verdict, "scopeNote": scope_note,
                    "reasoning": fields.get("reasoning", "")})
    print(json.dumps(fields, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
