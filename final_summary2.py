#!/usr/bin/env python3
"""
Regenerate the paper-level summary AGAIN after Claim 2's verdict changed from
verified -> inconclusive during the consistency-check follow-up. Same harness-
only rules as final_summary.py.

Run: ~/.venvs/icml26/bin/python final_summary2.py
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
    claim_results = session["claim_results"]
    total_model_minutes = 41.7 + 3  # + the claim2 reconsideration turns

    messages = [{"role": "system", "content": dow.SYSTEM_PROMPT}]
    summary_block = "\n\n".join(
        f"Claim {i+1}: \"{c['claim']}\"\n"
        f"  FINAL verdict: {c['verdict']}\n"
        f"  measured: {json.dumps(c['measuredMetrics'])}\n"
        f"  your finding/scopeNote: {c.get('finding','')} {c.get('scopeNote','')}"
        for i, c in enumerate(claim_results)
    )
    messages.append({"role": "user", "content": (
        "FINAL update: after finishing Claim 1 (5 rounds of scrutiny, final verdict falsified), "
        "you were also asked to reconsider Claim 2 given what you learned about hard-cut vs. "
        "Bayesian belief updates. You caught your own inconsistency (you had said the hard-cut "
        "implementation 'does not match the paper's algorithm' but still called it falsified) and "
        "correctly changed Claim 2's verdict to inconclusive, since a test that does not faithfully "
        "implement the paper's algorithm cannot establish either a verification or a falsification. "
        "Here is the FINAL state of both claims:\n\n"
        f"{summary_block}\n\n"
        f"Total model compute time across the entire session: approximately {total_model_minutes:.1f} "
        "minutes on an Apple M4 Max, CPU only, running you (qwen3:30b-a3b) locally via Ollama -- $0 "
        "cost, no GPU, no network access during any experiment.\n\n"
        "Write the FINAL paper-level summary reflecting BOTH final verdicts (Claim 1: falsified, "
        "Claim 2: inconclusive). Your executiveSummary and conclusion MUST state plainly that YOU "
        "(qwen3:30b-a3b, open-weights, local) were the reproducing agent for all code/interpretation/"
        "verdicts, that Claim 1 was revised across 5 scrutiny rounds, and that Claim 2 was honestly "
        "downgraded to inconclusive after you recognized your implementation did not faithfully match "
        "the paper's Bayesian PBA. Reply with a JSON object in a single ```json block with EXACTLY "
        "these keys:\n"
        '  "executiveSummary": 3-5 sentences, OUTCOME FIRST\n'
        '  "scope": one sentence describing the reduced scope relative to the full paper\n'
        '  "outcome": one short sentence, the bottom-line outcome\n'
        '  "conclusion": a paragraph covering both final verdicts, the scrutiny process on Claim 1, '
        "and why Claim 2 was honestly downgraded rather than left as a false verified\n"
        '  "links": a JSON array of source URLs actually used (fine to leave empty)'
    )})
    thinking, content = dow.call_model(messages, purpose="final_paper_summary2")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "note": "final summary2 JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the five keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="final_paper_summary2_retry")
        messages.append({"role": "assistant", "content": content})
        fields = dow.extract_json(content) or {}

    session["summary"] = fields
    session["total_model_minutes_all_rounds"] = total_model_minutes
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))
    dow.log_event({"type": "final_summary2_end", "summary": fields})
    print(json.dumps(fields, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
