"""
Item 5 audit (per DESIGN_FROZEN.md Section 9 statistical-unit discipline
and Section 10 endpoints). partition_seed is the independent statistical
unit (n=5); model seeds are NESTED within partition, never pooled as
independent observations. For each partition, the 3 nested model seeds
are summarized (mean) first; only THEN is consistency examined across
the 5 partitions -- the same two-level discipline as Item 4's seed-level
audit, one level deeper because Item 5 additionally nests seeds in
partitions.

Primary decisive metric: RRG. Secondary: strict/practical erosion,
conditional retention. Read-only; does not touch item5_raw.csv.
"""
import csv
from collections import defaultdict

RAW = "/Users/abanjar/All Project/UAV-FL/uav-federated-ids/repositories/paper-01-rare-class-erosion/paper1_rare_class_erosion/item5_cross_domain_validation/results/item5_raw.csv"
PARTITIONS = [101, 102, 103, 104, 105]
SEEDS = [11, 22, 33]
ROUNDS = [1, 5, 10, 15, 20]
CLASSES = ["Uploading_Attack", "Recon-PingSweep", "Backdoor_Malware", "XSS", "SqlInjection"]
ALGORITHMS = ["FedAvg", "FedNova"]

rows = list(csv.DictReader(open(RAW)))


def f(r, k):
    return float(r[k])


print("=" * 100)
print("PART 1 -- RRG (primary): partition-level mean (across 3 nested seeds), per algorithm/round/class")
print("          Consistency across the 5 partitions is the design's mandated comparison (n=5 units).")
print("=" * 100)
for algo in ALGORITHMS:
    for cls in CLASSES:
        for rnd in ROUNDS:
            partition_means = []
            for p in PARTITIONS:
                sub = [r for r in rows if r["algorithm"] == algo and r["class"] == cls
                       and r["round"] == str(rnd) and r["partition_seed"] == str(p)]
                vals = [f(r, "RRG") for r in sub]
                partition_means.append(sum(vals) / len(vals))
            pos = sum(1 for v in partition_means if v > 0)
            print(f"{algo:8s} {cls:18s} round={rnd:2d}  partition_means={['%.4f'%v for v in partition_means]}  "
                  f"(+:{pos}/5)")
    print()

print("=" * 100)
print("PART 2 -- Conditional retention: fraction of the 3 nested seeds retaining, per partition,")
print("          at round=20 (converged state) -- 0/3, 1/3, 2/3, or 3/3 per partition.")
print("=" * 100)
for algo in ALGORITHMS:
    for cls in CLASSES:
        line = [f"{algo:8s} {cls:18s} round=20"]
        for p in PARTITIONS:
            sub = [r for r in rows if r["algorithm"] == algo and r["class"] == cls
                   and r["round"] == "20" and r["partition_seed"] == str(p)]
            retained = sum(1 for r in sub if r["conditional_retention"] == "1")
            line.append(f"p{p}:{retained}/3")
        print("  ".join(line))
    print()

print("=" * 100)
print("PART 3 -- Strict/practical erosion: same fraction-of-3-seeds view, round=20")
print("=" * 100)
for algo in ALGORITHMS:
    for cls in CLASSES:
        line_s = [f"{algo:8s} {cls:18s} strict round=20"]
        line_p = [f"{algo:8s} {cls:18s} practical round=20"]
        for p in PARTITIONS:
            sub = [r for r in rows if r["algorithm"] == algo and r["class"] == cls
                   and r["round"] == "20" and r["partition_seed"] == str(p)]
            s_erosion = sum(1 for r in sub if r["strict_erosion"] == "1")
            p_erosion = sum(1 for r in sub if r["practical_erosion"] == "1")
            line_s.append(f"p{p}:{s_erosion}/3")
            line_p.append(f"p{p}:{p_erosion}/3")
        print("  ".join(line_s))
        print("  ".join(line_p))
    print()

print("=" * 100)
print("PART 4 -- Macro-F1 (context only), partition-level mean at round=20, per algorithm")
print("=" * 100)
for algo in ALGORITHMS:
    line = [f"{algo:8s} round=20"]
    for p in PARTITIONS:
        sub = [r for r in rows if r["algorithm"] == algo and r["round"] == "20"
               and r["partition_seed"] == str(p) and r["class"] == CLASSES[0]]  # macro_f1 same across classes/round
        vals = [f(r, "macro_f1_all_classes_validation") for r in sub]
        line.append(f"p{p}:{sum(vals)/len(vals):.4f}")
    print("  ".join(line))

print("\n" + "=" * 100)
print("PART 5 -- n_holders_this_draw range across partitions (context on holder-set size at alpha=0.1)")
print("=" * 100)
for cls in CLASSES:
    sub = [r for r in rows if r["class"] == cls and r["round"] == "1" and r["algorithm"] == "FedAvg"]
    holder_counts = sorted(set(int(r["n_holders_this_draw"]) for r in sub))
    print(f"  {cls:18s}: n_holders range across 5 partitions x 3 seeds = {holder_counts}")
