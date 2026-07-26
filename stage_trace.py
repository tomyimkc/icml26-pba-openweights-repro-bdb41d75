#!/usr/bin/env python3
"""Stage a public-safe trace.jsonl for this paper from the open-weights session trace.

Unlike the Claude-subagent papers (which pull a Claude transcript via
lib/traces.py:find_agent_transcripts), this paper's real "agent session" is
qwen3:30b-a3b's own recorded turns in orx-openweights-trace.jsonl (written by
driver_openweights.py + challenge_claim1.py). This script reuses the SAME
scrub()/build_trace() logic from lib/traces.py (imported, not modified) so the
public trace.jsonl follows the repo's existing convention: home paths -> ~,
secret-shaped tokens redacted, one JSON object per line.

Run: ~/.venvs/icml26/bin/python stage_trace.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
FACTORY = PAPER_DIR.parent.parent
sys.path.insert(0, str(FACTORY / "lib"))

from traces import build_trace  # noqa: E402  (reused, not modified)

SRC = PAPER_DIR / "orx-openweights-trace.jsonl"
DEST = PAPER_DIR / "trace.jsonl"


def main() -> int:
    if not SRC.is_file():
        print(f"missing source trace: {SRC}", file=sys.stderr)
        return 1
    info = build_trace(SRC, DEST)
    info.update({
        "orid": "Zlw4Kl5HEF",
        "trace": str(DEST),
        "source": str(SRC),
        "note": (
            "Open-weights reproduction: this is the FULL recorded session of "
            "qwen3:30b-a3b (via Ollama) as the reproducing agent, not a Claude "
            "subagent transcript. Produced by driver_openweights.py (initial "
            "claim1+claim2 write/verdict rounds) and challenge_claim1.py (the "
            "claim-1 falsification-scrutiny round). Scrubbed with the same "
            "lib/traces.py:scrub() used for the other papers in this factory."
        ),
    })
    (PAPER_DIR / "trace-receipt.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
