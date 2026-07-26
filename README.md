# Open-Weights Reproduction: Probabilistic Bisection Algorithm

Reproduction of one ICML 2026 submission for the **OpenResearch Open-Weights
Award**: ["Probabilistic Bisection Algorithm Provably Achieves Exponential
Convergence"](https://openreview.net) (orid `Zlw4Kl5HEF`, no public arXiv id
or code repo).

**Paper's claims tested:**

1. *"The probabilistic bisection algorithm (PBA) converges at a geometric
   rate, matching the performance of classical binary search under noiseless
   responses."*
2. *"PBA queries oscillate around the truth but steadily draw closer,
   yielding an estimator that rapidly concentrates on the truth."*

## Who did what — the model vs. the harness

**The reproducing agent was [`qwen3:30b-a3b`](https://ollama.com/library/qwen3),
an open-weights model served entirely locally via [Ollama](https://ollama.com)
on an Apple M4 Max — CPU only, no GPU, no network access, $0 cost.** The model
did all of the science:

- designed and wrote every experiment script from scratch (`experiments/exp_claim1.py`,
  `experiments/exp_claim2.py`) — there is no public repo for this paper, so the
  model implemented the probabilistic bisection algorithm itself, from its own
  knowledge and the claim text;
- decided the scale, seeds, and controls for each experiment;
- read its own scripts' `RESULT_JSON` output and assigned every verdict
  (`verified` / `falsified` / `toy` / `inconclusive`);
- revised its own work across five rounds of adversarial scrutiny on Claim 1,
  and caught and corrected its own internally-inconsistent verdict on Claim 2.

**The harness (`driver_openweights.py` + the `challenge_claim1*.py` /
`claim2_reconsider*.py` / `final_summary*.py` scripts) is plumbing only.** It:

- sent the model the paper claim + the experiment-script contract, extracted
  the code block the model wrote, and executed it verbatim (twice, to check
  determinism);
- fed stdout/stderr back to the model on failure and let the model decide the
  fix — the harness never edited the model's code or invented a number;
- for the scrutiny rounds, handed the model a specific technical concern (a
  bug in the noise model, an irreversible hard interval-cut instead of a
  genuine Bayesian belief update, a discretization floor, a root placed
  pathologically close to the algorithm's first query, a biased summary
  statistic) and let the model decide whether the concern was valid, whether
  to revise the script, and what the final verdict is;
- logged every prompt and response — including the model's own chain of
  thought where the API exposed it — to `orx-openweights-trace.jsonl`
  (published here, scrubbed, as `trace.jsonl`).

The harness never wrote an experiment, never computed a verdict, and never
supplied a number that didn't come from the model's own measured output.

## Final verdicts

| Claim | Verdict | Why |
|---|---|---|
| 1 — geometric convergence | **Falsified** | Survived 5 rounds of adversarial scrutiny (see below). Noiseless positive control (p=1.0) matches classical binary search exactly (ratio 0.5003, std 0.0 across 10 seeds); main case (p=0.9) measures 0.886 ± 0.15 — far from the 0.5 geometric rate the claim predicts; relaxed-precondition control (p=0.51) degrades as expected (0.9995). |
| 2 — oscillate-but-converge | **Inconclusive** | Not a failure of the claim — an implementation mismatch. The script used a hard interval-cut update instead of the paper's Bayesian belief-density update (the same category of bug fixed for Claim 1 in round 2 of scrutiny). The model caught its own contradiction (having called the implementation non-faithful to the paper, then still assigning `falsified`) and corrected the verdict to `inconclusive`. |

Full reasoning, measured numbers, and scope notes: `results.json` (executive
summary, per-claim method/finding/scopeNote) and `measured.json` (raw
`RESULT_JSON` metrics from each script).

### The five-round scrutiny process on Claim 1

Claim 1's script went through five rounds of harness-initiated challenges,
each handing the model one concrete, correctly-diagnosed technical concern and
letting the model decide whether/how to respond:

1. **Round 1** — the oracle's correctness probability decayed toward 0.5 near
   the root instead of staying fixed at `p`, unlike the paper's fixed-`p`
   oracle.
2. **Round 2** — the update rule was a hard interval bisection (irreversible
   on one wrong response), not the paper's actual Bayesian belief-density
   update, which never drives any point's probability to exactly zero.
3. **Round 3** — a fixed discretization grid (`N=10001` bins) caused the
   belief to floor out after ~14 of 100 steps, biasing every ratio toward 1,
   including the positive control. Fixed with an exact, non-discretized
   piecewise-constant belief representation.
4. **Round 4** — the root was placed at `0.5 + 1e-10`, i.e. essentially
   exactly at the algorithm's own first query point (the median of a uniform
   prior), so the "error" started at float64 noise floor with nothing left to
   converge over.
5. **Round 5 (final)** — the summary statistic itself (arithmetic mean of raw
   per-step error ratios) was high-variance and upward-biased on a process
   that legitimately oscillates while converging (exactly what the paper's
   Claim 2 describes). Replaced with a log-linear decay slope across 10
   seeds, which is what produced the final falsification.

This process is fully recorded in the harness scripts themselves (each
docstring explains the concern it hands to the model) and in `trace.jsonl`.

## Repository layout

```
paper.json                        paper metadata (orid, claims, abstract)
experiments/exp_claim1.py         model-authored experiment for Claim 1 (final version)
experiments/exp_claim2.py         model-authored experiment for Claim 2 (final version)
driver_openweights.py             harness: initial write/execute/verdict loop for both claims
challenge_claim1.py               harness: Claim 1 scrutiny round 1
challenge_claim1_round2.py        harness: Claim 1 scrutiny round 2
challenge_claim1_round3.py        harness: Claim 1 scrutiny round 3
challenge_claim1_round4.py        harness: Claim 1 scrutiny round 4
challenge_claim1_round5.py        harness: Claim 1 scrutiny round 5 (final)
claim2_reconsider.py              harness: asks the model to reconsider Claim 2 given what round 2 established
claim2_reconsider2.py             harness: points out the model's internal inconsistency on Claim 2's verdict
claim2_posterfinding.py           harness: small follow-up to keep Claim 2's poster summary consistent with its final verdict
final_summary.py                  harness: paper-level summary turn after Claim 1's scrutiny concluded
final_summary2.py                 harness: regenerates the paper-level summary after Claim 2's verdict changed
stage_trace.py                    internal tool that scrubs the raw session trace for public release (depends on
                                   this factory's shared `lib/traces.py`, not included in this repo)
results.json                      final report: executive summary, per-claim method/finding/scopeNote/verdict
measured.json                     raw RESULT_JSON metrics printed by each experiment script
openweights_session_output.json   full structured output of the initial driver_openweights.py session
build-receipt.json                build metadata (validated: true, claim count, verdicts)
trace-receipt.json                metadata about the trace scrub (event count, byte counts)
trace.jsonl                       full recorded session trace of qwen3:30b-a3b (94 events; scrubbed of local
                                   paths and secret-shaped strings) — every prompt sent and every response
                                   received, across the initial run and all scrutiny rounds
```

## How to re-run this reproduction

Requirements: Python 3 with `numpy`, and [Ollama](https://ollama.com) running
locally with the `qwen3:30b-a3b` model pulled (`ollama pull qwen3:30b-a3b`).
No GPU and no network access are used or required — everything runs on CPU.

```sh
# 1. Pull the open-weights model once:
ollama pull qwen3:30b-a3b

# 2. Run the initial two-claim write/execute/verdict loop. This writes
#    experiments/exp_claim1.py and experiments/exp_claim2.py from scratch,
#    executes them, and has the model assign initial verdicts + a paper
#    summary. Logs every turn to orx-openweights-trace.jsonl.
python3 driver_openweights.py

# 3. Run the five scrutiny rounds on Claim 1, in order (each reuses the
#    previous round's script and appends to the same trace file):
python3 challenge_claim1.py
python3 challenge_claim1_round2.py
python3 challenge_claim1_round3.py
python3 challenge_claim1_round4.py
python3 challenge_claim1_round5.py

# 4. Reconcile Claim 2's verdict given what the scrutiny rounds established:
python3 claim2_reconsider.py
python3 claim2_reconsider2.py
python3 claim2_posterfinding.py

# 5. Regenerate the final paper-level summary reflecting both final verdicts:
python3 final_summary.py
python3 final_summary2.py
```

Each experiment script (`experiments/exp_claim*.py`) is independently
runnable and deterministic (seeded), and is executed twice by the harness on
every write to confirm determinism before its output is trusted. Because
`qwen3:30b-a3b` is not deterministic across model calls (`temperature=0.3`),
re-running the full harness end-to-end may produce a different exploration
path than the one recorded in `trace.jsonl`, though the underlying
mathematics of PBA — and therefore the measured convergence behavior once a
faithful implementation is reached — is not expected to change.

## Honesty / scope notes

- No arXiv id or public code repository was available for this paper; the
  model implemented PBA from its own knowledge, which is why the first four
  rounds of scrutiny surfaced (and fixed) real implementation bugs before
  reaching a metric the model could trust.
- Only symmetric, one-dimensional root-finding under uniform noise was
  tested; non-uniform noise distributions and higher-dimensional settings
  were out of scope.
- This reproduction is a **record of a completed, already-run session** —
  publishing it here does not re-run any compute.
