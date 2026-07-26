#!/usr/bin/env python3
"""Small follow-up: Claim 2's posterFinding still reads "...verifying claim",
left over from before the verdict was honestly downgraded to inconclusive.
Ask the model for a consistent one-liner. Harness only.

Run: ~/.venvs/icml26/bin/python claim2_posterfinding.py
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
    c2 = session["claim_results"][1]

    messages = [{"role": "system", "content": dow.SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": f"""Your Claim 2 posterFinding still reads: "{c2['posterFinding']}" -- \
this says "verifying claim" but your final verdict is "{c2['verdict']}", not verified. This is now \
inconsistent and would be misleading on the poster card. Give a corrected posterFinding (<=200 chars) \
consistent with your final inconclusive verdict and scope note: "{c2['scopeNote']}". It should still \
report the measured numbers (they are real measurements, just not evidence for or against the claim \
given the implementation mismatch you identified). Reply with a JSON object in a single ```json block \
with EXACTLY one key: "posterFinding" (<=200 chars).
"""})
    thinking, content = dow.call_model(messages, purpose="claim2_posterfinding")
    messages.append({"role": "assistant", "content": content})
    fields = dow.extract_json(content) or {}
    pf = (fields.get("posterFinding") or c2["posterFinding"])[:200]
    session["claim_results"][1]["posterFinding"] = pf
    SESSION_OUT_PATH.write_text(json.dumps(session, indent=2))
    dow.log_event({"type": "claim2_posterfinding_end", "posterFinding": pf})
    print(json.dumps({"posterFinding": pf}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
