"""Aggregate LODO fold metrics and evaluate the pre-registered success criteria.

Reads every metrics.json under a run directory, reports mean +- std across seeds,
runs a paired permutation test on the coverage gap, and writes summary.json. The
thresholds are fixed constants rather than arguments.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.significance import paired_permutation_test

# Pre-registered thresholds (PROTOCOL.md §8).
GATE_MAX_COVERAGE_GAP = 0.05   # <= 5 percentage points at each nominal level
GATE_MAX_WIDTH_INFLATION = 2.0  # <= 2x the in-domain MPIW
GATE_MAX_MAPE_RATIO = 1.10     # within ~10% relative of the tuned baseline
BASELINE = "split_cp"
PROPOSED = "physics_weighted_cp"


def _load(run: Path) -> list[dict]:
    recs = [json.loads(p.read_text()) for p in sorted(run.rglob("metrics.json"))]
    if not recs:
        raise SystemExit(f"no metrics.json found under {run}")
    return recs


def _by(recs, key):
    out = defaultdict(list)
    for r in recs:
        out[r[key]].append(r)
    return out


def _interval_rows(rec):
    """Yield (variant, nominal, metrics dict) for one fold record."""
    for variant, entries in rec["intervals"].items():
        for m in entries:
            yield variant, float(m["nominal"]), m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="summary.json path (default <run>/summary.json)")
    ap.add_argument("--baseline-mape", type=float, default=None,
                    help="tuned baseline's mean target MAPE (fraction), for PROTOCOL "
                         "§8.4. Without it criterion 4 is reported NOT EVALUABLE rather "
                         "than assumed to pass.")
    args = ap.parse_args()
    run = Path(args.run)
    recs = _load(run)

    seeds = sorted({r["seed"] for r in recs})
    domains = sorted({r["held_out_domain"] for r in recs})
    nominals = sorted({n for r in recs for _, n, _ in _interval_rows(r)})
    print(f"run: {run}")
    print(f"  {len(recs)} fold-runs | {len(seeds)} seeds {seeds} | "
          f"{len(domains)} held-out domains")

    print("\n=== point accuracy (MAPE %, mean +- std over seeds) ===")
    print(f"{'held-out domain':16s} {'n':>4s} {'MAPE %':>16s} {'clipped':>8s}")
    point = {}
    for dom, rs in sorted(_by(recs, "held_out_domain").items()):
        m = np.array([r["point_target"]["mape"] for r in rs]) * 100
        clip = np.mean([r.get("n_predictions_clipped", 0) for r in rs])
        point[dom] = m
        print(f"{dom:16s} {rs[0]['n_target_cells']:>4d} "
              f"{m.mean():>9.2f} +-{m.std():5.2f} {clip:>8.1f}")
    allm = np.concatenate(list(point.values()))
    print(f"{'ALL':16s} {'':>4s} {allm.mean():>9.2f} +-{allm.std():5.2f}")

    summary = {"run": str(run), "seeds": seeds, "domains": domains,
               "nominals": nominals, "coverage": {}, "gate": {}}
    for nominal in nominals:
        print(f"\n=== coverage at nominal {nominal:.2f} "
              f"(PICP mean +- std; gap = |PICP - nominal|) ===")
        print(f"{'held-out domain':16s} {'split_cp':>18s} {'physics_wtd':>18s} "
              f"{'MPIW ratio':>11s} {'N_eff':>8s}")
        per_dom = {}
        for dom, rs in sorted(_by(recs, "held_out_domain").items()):
            got = {}
            for variant in (BASELINE, PROPOSED):
                cov, w, ne = [], [], []
                for r in rs:
                    for v, n, m in _interval_rows(r):
                        if v == variant and abs(n - nominal) < 1e-9:
                            cov.append(m["picp"]); w.append(m["mpiw"])
                            ne.append(m.get("diag_n_eff", np.nan))
                got[variant] = (np.array(cov), np.array(w), np.array(ne))
            b_cov, b_w, _ = got[BASELINE]
            p_cov, p_w, p_ne = got[PROPOSED]
            ratio = float(np.mean(p_w) / np.mean(b_w)) if np.mean(b_w) > 0 else np.nan
            per_dom[dom] = {
                "split_cp_picp_mean": float(b_cov.mean()), "split_cp_picp_std": float(b_cov.std()),
                "physics_picp_mean": float(p_cov.mean()), "physics_picp_std": float(p_cov.std()),
                "mpiw_ratio_physics_over_split": ratio,
                "n_eff_mean": float(np.nanmean(p_ne)),
            }
            print(f"{dom:16s} {b_cov.mean():>10.3f} +-{b_cov.std():5.3f} "
                  f"{p_cov.mean():>10.3f} +-{p_cov.std():5.3f} "
                  f"{ratio:>11.2f} {np.nanmean(p_ne):>8.1f}")
        summary["coverage"][f"{nominal:.2f}"] = per_dom

        # Pair by held-out domain, the independent unit: all seeds of a domain score
        # the same target cells, so seeds are averaged within a domain first.
        gap_b, gap_p = defaultdict(list), defaultdict(list)
        for r in recs:
            for v, n, m in _interval_rows(r):
                if abs(n - nominal) > 1e-9:
                    continue
                if v == BASELINE:
                    gap_b[r["held_out_domain"]].append(m["coverage_gap"])
                elif v == PROPOSED:
                    gap_p[r["held_out_domain"]].append(m["coverage_gap"])
        doms_paired = sorted(set(gap_b) & set(gap_p))
        pairs_b = [float(np.mean(gap_b[d])) for d in doms_paired]
        pairs_p = [float(np.mean(gap_p[d])) for d in doms_paired]
        diff, p = paired_permutation_test(np.array(pairs_p), np.array(pairs_b))
        print(f"  paired test on coverage gap (proposed - baseline): "
              f"mean diff {diff:+.4f}, p = {p:.4f}"
              f"  -> {'significant' if p < 0.05 else 'NOT significant'}")
        summary["coverage"][f"{nominal:.2f}"]["_paired_test"] = {
            "mean_gap_diff": diff, "p_value": p, "n_pairs": len(pairs_b),
            "pairing_unit": "held_out_domain"}

    print("\n=== KILL GATE (PROTOCOL.md §8, thresholds signed off pre-run) ===")
    verdicts = {}
    for nominal in nominals:
        per_dom = summary["coverage"][f"{nominal:.2f}"]
        best_gap, best_dom = np.inf, None
        for dom, v in per_dom.items():
            if dom.startswith("_"):
                continue
            gap = abs(v["physics_picp_mean"] - nominal)
            if gap < best_gap:
                best_gap, best_dom = gap, dom
        verdicts[nominal] = (best_gap, best_dom)
        ok = best_gap <= GATE_MAX_COVERAGE_GAP
        print(f"  criterion 1 @ {nominal:.2f}: best coverage gap {best_gap:.3f} "
              f"({best_dom}) vs threshold {GATE_MAX_COVERAGE_GAP:.2f}  "
              f"-> {'PASS' if ok else 'FAIL'}")

    c1 = all(v[0] <= GATE_MAX_COVERAGE_GAP for v in verdicts.values())

    # Criterion 2 compares interval width with the in-domain MPIW, not with the
    # baseline method's width.
    have_in_domain = any("intervals_in_domain" in r for r in recs)
    if not have_in_domain:
        c2 = None
        print("  criterion 2: CANNOT BE EVALUATED -- no in-domain reference arm in "
              "this run.\n"
              "               PROTOCOL.md §8 defines width inflation relative to the\n"
              "               in-domain MPIW. Re-run with the in-domain arm before\n"
              "               treating any gate verdict as real.")
    else:
        infl = []
        for r in recs:
            in_by_nom = {float(m["nominal"]): m["mpiw"]
                         for m in r.get("intervals_in_domain", [])}
            for v, n, m in _interval_rows(r):
                if v == PROPOSED and n in in_by_nom and in_by_nom[n] > 0:
                    infl.append(m["mpiw"] / in_by_nom[n])
        c2 = float(np.nanmax(infl)) <= GATE_MAX_WIDTH_INFLATION
        print(f"  criterion 2: max MPIW inflation vs IN-DOMAIN "
              f"{np.nanmax(infl):.2f} vs threshold {GATE_MAX_WIDTH_INFLATION:.1f}"
              f"  -> {'PASS' if c2 else 'FAIL'}")
        summary["gate"]["max_width_inflation_vs_in_domain"] = float(np.nanmax(infl))

    c3 = all(summary["coverage"][f"{n:.2f}"]["_paired_test"]["p_value"] < 0.05
             for n in nominals)
    print(f"  criterion 3: paired test significant at every nominal -> "
          f"{'PASS' if c3 else 'FAIL'}")

    # Criterion 4 compares this run's MAPE with a separate tuned-baseline run, so it
    # is evaluable only when that baseline MAPE is supplied.
    baseline_mape = getattr(args, "baseline_mape", None)
    mape = float(np.mean([r["point_target"]["mape"] for r in recs]))
    summary["gate"]["target_mape"] = mape
    if baseline_mape is None:
        c4 = None
        print(f"  criterion 4: NOT EVALUABLE -- needs the tuned baseline's target MAPE "
              f"for comparison (this run: {mape * 100:.1f}%). Pass --baseline-mape, or "
              f"aggregate the tuned arm first.")
    else:
        ratio = mape / float(baseline_mape)
        c4 = ratio <= GATE_MAX_MAPE_RATIO
        summary["gate"]["mape_ratio_vs_baseline"] = ratio
        print(f"  criterion 4: target MAPE {mape * 100:.1f}% vs tuned baseline "
              f"{float(baseline_mape) * 100:.1f}% = {ratio:.2f}x vs threshold "
              f"{GATE_MAX_MAPE_RATIO:.2f}x -> {'PASS' if c4 else 'FAIL'}")

    overall = bool(c1) and (c2 is True) and c3 and (c4 is True)
    if c2 is None:
        print(f"\n  GATE VERDICT: INCONCLUSIVE -- criterion 2 not evaluable "
              f"(criteria 1/3/4 = {c1}/{c3}/{c4})")
    else:
        print(f"\n  GATE VERDICT: {'PASS' if overall else 'FAIL'}"
              f"   (criteria 1/2/3/4 = {c1}/{c2}/{c3}/{c4})")
    if not overall:
        print("  Per the Gate-1 risk stance, a FAIL means returning to Gate 1 and\n"
              "  re-ideating rather than writing up a negative result. This script\n"
              "  reports; the decision is the user's.")
    summary["gate"].update({
        "criterion_1_coverage_gap": bool(c1),
        "criterion_2_width_inflation": None if c2 is None else bool(c2),
        "criterion_3_significance": bool(c3),
        "criterion_4_point_accuracy": None if c4 is None else bool(c4),
        "verdict": ("INCONCLUSIVE" if (c2 is None or c4 is None)
                    else ("PASS" if overall else "FAIL")),
        "thresholds": {"max_coverage_gap": GATE_MAX_COVERAGE_GAP,
                       "max_width_inflation": GATE_MAX_WIDTH_INFLATION,
                       "max_mape_ratio": GATE_MAX_MAPE_RATIO},
    })

    dest = Path(args.out) if args.out else run / "summary.json"
    dest.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
