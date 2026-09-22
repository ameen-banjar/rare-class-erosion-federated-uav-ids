"""
Item 4 audit (per DESIGN_FROZEN.md's interpretation matrix, decisive
metrics only: Conditional Retention + Delta-Margin; seed as the
independent statistical unit; per-round reporting since Item 4's scope
is the proximal effect of one aggregation step at a given round, not a
pooled long-horizon claim).

Read-only: does not touch item4_raw.csv or item4_determinism_gate.csv.
"""
import csv
from collections import defaultdict

RAW = "/Users/abanjar/All Project/UAV-FL/uav-federated-ids/repositories/paper-01-rare-class-erosion/paper1_rare_class_erosion/item4_mechanistic_localization/results/item4_raw.csv"
SEEDS = [11, 22, 33, 44, 55]
ROUNDS = [1, 5, 10, 20, 30, 45]
CLASSES = ["Manipulation", "Replay"]
INTERVENTIONS = ["A", "B", "C"]

rows = list(csv.DictReader(open(RAW)))

def f(r, k):
    return float(r[k])

print("=" * 100)
print("PART 1 -- Event-count view (holder x seed instances), matching manuscript's own")
print("          '0/43'-style convention. Purely descriptive.")
print("=" * 100)
for rnd in ROUNDS:
    for cls in CLASSES:
        line = [f"round={rnd:2d} class={cls:12s}"]
        for interv in ["0"] + INTERVENTIONS:
            sub = [r for r in rows if r["round"] == str(rnd) and r["target_class"] == cls
                   and r["intervention"] == interv]
            n = len(sub)
            retained = sum(1 for r in sub if r["conditional_retention"] == "1")
            line.append(f"{interv}:{retained}/{n}")
        print("  ".join(line))
    print()

print("=" * 100)
print("PART 2 -- Seed-level consistency (the design's mandated independent unit).")
print("          A seed 'shows rescue' under an intervention/class/round if AT LEAST ONE")
print("          of that class's holders has conditional_retention=1 (i.e. the holder-level")
print("          floor used elsewhere in the manuscript: 'best holder'). ANY-holder is the")
print("          same floor DESIGN_FROZEN.md's own metric-1 definition uses")
print("          ('local_recall_holder>0 (الحد الأدنى: أفضل حامل)').")
print("=" * 100)
for rnd in ROUNDS:
    for cls in CLASSES:
        line = [f"round={rnd:2d} class={cls:12s}"]
        for interv in ["0"] + INTERVENTIONS:
            seeds_rescued = 0
            for seed in SEEDS:
                sub = [r for r in rows if r["round"] == str(rnd) and r["target_class"] == cls
                       and r["intervention"] == interv and r["seed"] == str(seed)]
                if any(r["conditional_retention"] == "1" for r in sub):
                    seeds_rescued += 1
            line.append(f"{interv}:{seeds_rescued}/5")
        print("  ".join(line))
    print()

print("=" * 100)
print("PART 3 -- Delta-Margin (primary continuous metric), mean +/- across the class's")
print("          holders, per seed, per round. Sign consistency across the 5 seeds is")
print("          what DESIGN_FROZEN.md asks for (RQ3-style direction count), not p-values.")
print("=" * 100)
for rnd in ROUNDS:
    for cls in CLASSES:
        for interv in INTERVENTIONS:
            seed_means = []
            for seed in SEEDS:
                sub = [r for r in rows if r["round"] == str(rnd) and r["target_class"] == cls
                       and r["intervention"] == interv and r["seed"] == str(seed)]
                vals = [f(r, "delta_margin") for r in sub]
                seed_means.append(sum(vals) / len(vals))
            pos = sum(1 for v in seed_means if v > 0)
            neg = sum(1 for v in seed_means if v < 0)
            zero = sum(1 for v in seed_means if v == 0)
            print(f"round={rnd:2d} class={cls:12s} interv={interv}  "
                  f"seed_means={['%.4f' % v for v in seed_means]}  "
                  f"(+:{pos} -:{neg} 0:{zero})")
    print()

print("=" * 100)
print("PART 4 -- Intervention-C validity flag (macro-F1 collapse >15pp vs intervention 0),")
print("          per DESIGN_FROZEN.md's pre-registered exclusion rule for C.")
print("=" * 100)
for rnd in ROUNDS:
    for cls in CLASSES:
        for seed in SEEDS:
            sub = [r for r in rows if r["round"] == str(rnd) and r["target_class"] == cls
                   and r["intervention"] == "C" and r["seed"] == str(seed)]
            if sub and sub[0]["intervention_c_flagged_invalid"] == "True":
                print(f"  round={rnd} class={cls} seed={seed}: C FLAGGED INVALID "
                      f"(macro_f1_c={sub[0]['intervention_c_macro_f1_all_classes']})")
