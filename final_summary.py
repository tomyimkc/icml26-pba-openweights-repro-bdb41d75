#!/usr/bin/env python3
"""
Final paper-level summary turn, run AFTER all scrutiny rounds on claim 1.
Harness only: assembles context, asks qwen3:30b-a3b to write the paper-level
executiveSummary/scope/outcome/conclusion given BOTH final claim results and
an honest account of the 5-round scrutiny process on claim 1. The model
decides the wording; the harness only supplies the facts (verdicts, measured
numbers, round count, hardware/time).

Run: ~/.venvs/icml26/bin/python final_summary.py
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

    total_model_minutes = 41.7  # sum of measured model wall_seconds across the whole session
    n_scrutiny_rounds = 5

    messages = [{"role": "system", "content": dow.SYSTEM_PROMPT}]
    summary_block = "\n\n".join(
        f"Claim {i+1}: \"{c['claim']}\"\n"
        f"  FINAL verdict: {c['verdict']}\n"
        f"  measured: {json.dumps(c['measuredMetrics'])}\n"
        f"  your finding: {c['finding']}"
        for i, c in enumerate(claim_results)
    )
    messages.append({"role": "user", "content": (
        "You have now completed both claims of this reproduction, including 5 rounds of "
        "scrutiny/self-revision on Claim 1 (you were shown specific technical concerns about "
        "your Claim 1 script after each attempt -- an incorrect noise model, a hard-cut vs. "
        "Bayesian update issue, a discretization floor, a root placed too close to your "
        "algorithm's first query, and finally a wrong summary statistic combined with a "
        "single-seed sample size -- and you decided each time whether to revise and what the "
        "verdict should be). Your Claim 1 verdict changed from falsified -> falsified (revised "
        "reasoning) -> inconclusive -> falsified -> falsified (final, now with a clean, "
        "validated positive control, a properly-degrading precondition-relaxed control, and "
        "10-seed evidence with reported spread). Here is the FINAL summary of both claims:\n\n"
        f"{summary_block}\n\n"
        f"Total measured model compute time across the ENTIRE session (initial two claims plus "
        f"all 5 scrutiny rounds on claim 1): approximately {total_model_minutes:.1f} minutes on "
        "an Apple M4 Max, CPU only, running you (qwen3:30b-a3b) locally via Ollama -- $0 cost, no "
        "GPU, no network access during any experiment.\n\n"
        "Write the paper-level summary. Your executiveSummary and conclusion MUST state plainly "
        "that YOU (an open-weights model, qwen3:30b-a3b) were the reproducing agent that wrote "
        "the experiment code, ran it, interpreted the results, and assigned every verdict, "
        "including revising Claim 1 across 5 rounds of scrutiny before reaching your final "
        "verdict -- and that a harness executed your code and recorded the process but made no "
        "scientific decisions. Reply with a JSON object in a single ```json block with EXACTLY "
        "these keys:\n"
        '  "executiveSummary": 3-5 sentences, OUTCOME FIRST: what reproduced, what did not, '
        "what exactly was tested vs the paper's full setup, and the hardware/time/cost, AND "
        "that you are an open-weights local model that revised Claim 1 through several rounds "
        "of scrutiny before finalizing your verdict\n"
        '  "scope": one sentence describing the reduced scope relative to the full paper\n'
        '  "outcome": one short sentence, the bottom-line outcome\n'
        '  "conclusion": a paragraph: which claims were supported/falsified/inconclusive, the '
        "honest reproducibility notes, and a brief, honest account of how Claim 1's verdict "
        "evolved across the scrutiny rounds and why you trust the final one\n"
        '  "links": a JSON array of any source URLs you genuinely used or want to cite '
        "(fine to leave empty)"
    )})
    thinking, content = dow.call_model(messages, purpose="final_paper_summary")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content)
    if not fields:
        dow.log_event({"type": "harness_note", "note": "final summary JSON unparseable; retrying once"})
        messages.append({"role": "user", "content": (
            "I could not parse a JSON object from your last reply. Please reply again with "
            "ONLY a single ```json block containing exactly the five keys requested."
        )})
        thinking, content = dow.call_model(messages, purpose="final_paper_summary_retry")
        messages.append({"role": "assistant", "content": content})
        fields = dow.extract_json(content) or {}

    session["summary"] = fields
    session["total_model_minutes_all_rounds"] = total_model_minutes
    session["n_scrutiny_rounds_claim1"] = n_scrutiny_rounds
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))
    dow.log_event({"type": "final_summary_end", "summary": fields})
    print(json.dumps(fields, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
