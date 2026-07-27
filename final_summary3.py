#!/usr/bin/env python3
"""
Regenerate the paper-level summary AGAIN after an independent judge scored
Claim 2 as inconclusive (0 points), pointing out the hard-cut-vs-Bayesian gap
that Claim 2's `inconclusive` verdict already conceded, and Claim 2 was then
carried through three further rounds (challenge_claim2_round1/2/3.py) that
implemented a genuine Bayesian PBA, fixed a root-placement/grid-resolution
stall, added a real oscillation measurement, and reached a new FINAL verdict.
Same harness-only rules as final_summary.py / final_summary2.py: the harness
assembles the model's own JSON fields into results.json verbatim, and invents
no content or numbers.

Run: /Users/tom/.venvs/icml26/bin/python final_summary3.py
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
RESULTS_PATH = PAPER_DIR / "results.json"

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
    claim_results = session["claim_results"]

    messages = [{"role": "system", "content": dow.SYSTEM_PROMPT}]
    summary_block = "\n\n".join(
        f"Claim {i+1}: \"{c['claim']}\"\n"
        f"  FINAL verdict: {c['verdict']}\n"
        f"  measured: {json.dumps(c['measuredMetrics'])}\n"
        f"  your finding/scopeNote: {c.get('finding','')} {c.get('scopeNote','')}"
        for i, c in enumerate(claim_results)
    )
    messages.append({"role": "user", "content": (
        "ANOTHER FINAL update. An independent judge reviewed the reproduction and scored Claim 2 "
        f"(then inconclusive) as 0 points, with this verbatim reasoning:\n\n  \"{JUDGE_FEEDBACK}\"\n\n"
        "This confirmed the gap you had already found yourself, and the harness then drove Claim 2 "
        "through three further scrutiny rounds: round 1 replaced the hard-cut script with a genuine "
        "Bayesian PBA (posterior belief density, multiplicative likelihood update, query at the "
        "posterior median); round 2 fixed a root-placement + grid-resolution stall the harness found "
        "by independent verification; round 3 added a direct oscillation measurement (sign changes) "
        "alongside the concentration/decay-rate measurement, since the claim asserts both, and the "
        "harness had flagged that only concentration was being tested. Claim 2's FINAL verdict is now "
        "verified. Claim 1's verdict is unchanged from before (falsified, after 5 earlier scrutiny "
        "rounds). Here is the FINAL state of both claims:\n\n"
        f"{summary_block}\n\n"
        "Write the FINAL paper-level summary reflecting BOTH final verdicts (Claim 1: falsified, "
        "Claim 2: verified). Your executiveSummary and conclusion MUST state plainly that YOU "
        "(qwen3:30b-a3b, open-weights, local) were the reproducing agent for all code/interpretation/"
        "verdicts, that Claim 1 was revised across 5 scrutiny rounds, and that Claim 2 went through an "
        "honest inconclusive verdict (correctly identifying its own hard-cut implementation did not "
        "match the paper's algorithm), was confirmed as such by an independent judge, and was then "
        "carried through 3 further rounds that implemented the real Bayesian PBA and reached a "
        "genuinely verified conclusion. Reply with a JSON object in a single ```json block with "
        "EXACTLY these keys:\n"
        '  "executiveSummary": 3-5 sentences, OUTCOME FIRST\n'
        '  "scope": one sentence describing the reduced scope relative to the full paper\n'
        '  "outcome": one short sentence, the bottom-line outcome\n'
        '  "conclusion": a paragraph covering both final verdicts, the scrutiny process on both '
        "claims, and how Claim 2 went from a correct honest inconclusive to a genuinely verified "
        "result once the real algorithm was implemented\n"
        '  "links": a JSON array of source URLs actually used (fine to leave empty)'
    )})
    thinking, content = dow.call_model(messages, purpose="final_paper_summary3")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "note": "final summary3 JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the five keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="final_paper_summary3_retry")
        messages.append({"role": "assistant", "content": content})
        fields = dow.extract_json(content) or {}

    session["summary"] = fields
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))
    dow.log_event({"type": "final_summary3_end", "summary": fields})

    # Assemble results.json's top-level fields from the model's own text, verbatim (harness
    # plumbing only -- no content invented here).
    results = json.loads(RESULTS_PATH.read_text())
    if fields.get("executiveSummary"):
        results["executiveSummary"] = fields["executiveSummary"]
    if fields.get("outcome") or fields.get("scope"):
        results.setdefault("scopeCost", {})
        if fields.get("scope"):
            results["scopeCost"]["scope"] = fields["scope"]
        if fields.get("outcome"):
            results["scopeCost"]["outcome"] = fields["outcome"]
    if fields.get("conclusion"):
        results["conclusion"] = fields["conclusion"]
    if isinstance(fields.get("links"), list):
        results["links"] = fields["links"]
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")

    print(json.dumps(fields, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
