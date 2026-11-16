"""Final simulation for the capstone project PG-S2-49.

Simulates phylogenetic trees with three time intervals under the birth-death
sampling model with piecewise constant rates (Phylogenie 3.10.0). Each interval
has its own birth rate (lambda) and death rate (mu), and the rates change at two
random change times. Two sets are made:

    free set       every lambda and mu is drawn independently
    different set  lambda_2 is drawn first, then lambda_1 and lambda_3 are
                   drawn so they are more than 2x lambda_2 or less than half of it

A tree is kept only if its end time (from the origin) is at least
change_time_2 + 1, so the third interval has enough events.

Usage (Linux or WSL):
    python simulation.py                     # final run: seed 70, 10000 trees per set, 500 tips
    python simulation.py 69 --n-free 20 --n-different 20 --tag test   # quick test

Output, for each set, in outputs/<name>-kept/ :
    data/<id>.nwk     the kept trees (Newick)
    metadata.csv      true values of each tree and its end time
    y.npy             true values as an array (column order in y_columns.txt)
    file_ids.npy      tree ids in the same order as y.npy
    <name>.yaml       the Phylogenie settings used
    run_info.txt      short summary of the run
plus a zip file <name>-kept.zip of that folder.
"""
import os
import sys
import time
import shutil
import argparse
import subprocess
import yaml
import numpy as np
import pandas as pd
from phylogenie import load_newick

# ---------------- settings
parser = argparse.ArgumentParser(description="Three interval birth-death sampling simulation")
parser.add_argument("seed", type=int, nargs="?", default=70, help="random seed (final run: 70)")
parser.add_argument("--n-free", type=int, default=10000, help="trees to simulate in the free set")
parser.add_argument("--n-different", type=int, default=10000, help="trees to simulate in the different set")
parser.add_argument("--n-jobs", type=int, default=-1, help="CPU cores to use, -1 uses all cores")
parser.add_argument("--n-leaves", type=int, default=500, help="tips per tree")
parser.add_argument("--timeout", type=float, default=1.0, help="seconds allowed per tree")
parser.add_argument("--lambda-min", type=float, default=0.1)
parser.add_argument("--lambda-max", type=float, default=1.0)
parser.add_argument("--mu-min", type=float, default=0.1)
parser.add_argument("--mu-max", type=float, default=1.0)
parser.add_argument("--min-gap", type=float, default=1.0, help="smallest gap between the two change times")
parser.add_argument("--tag", type=str, default="500tips", help="label used in the output names")
args = parser.parse_args()

CHANGE_RANGE = (2.0, 8.0)      # both change times fall in this range
MAX_TIME = 10.0                # largest tree height allowed by Phylogenie
SURVIVE_AFTER_CHANGE = 1.0     # filter: end time >= change_time_2 + 1
LAMBDA_RANGE = (args.lambda_min, args.lambda_max)
MU_RANGE = (args.mu_min, args.mu_max)
PSI = 1.0                      # sampling rate, fixed
N_LEAVES = args.n_leaves
TIMEOUT = args.timeout
N_JOBS = args.n_jobs
N_TREES = {"free": args.n_free, "different": args.n_different}
SEED = args.seed
SETS = ["free", "different"]
TARGETS = ["change_time_1", "change_time_2", "lambda_1", "lambda_2", "lambda_3", "mu_1", "mu_2", "mu_3"]


# ---------------- Phylogenie settings for one set
def make_config(set_name, output_dir, n_samples):
    lo, hi = LAMBDA_RANGE
    span = (CHANGE_RANGE[1] - CHANGE_RANGE[0]) - args.min_gap
    # change times: a and b drawn from U(0, 5)
    # change_time_1 = 2 + min(a, b), change_time_2 = 3 + max(a, b)
    ctx = {
        "_a": {"type": "uniform", "low": 0.0, "high": span},
        "_b": {"type": "uniform", "low": 0.0, "high": span},
        "change_time_1": f"{CHANGE_RANGE[0]} + np.minimum(_a, _b)",
        "change_time_2": f"{CHANGE_RANGE[0] + args.min_gap} + np.maximum(_a, _b)",
    }
    if set_name == "free":
        # free set: each lambda drawn independently
        for k in (1, 2, 3):
            ctx[f"lambda_{k}"] = {"type": "uniform", "low": lo, "high": hi}
    else:
        # different set: lambda_2 first, then lambda_1 and lambda_3 are each
        # drawn from U(2 * lambda_2, 1) (up) or U(0.1, lambda_2 / 2) (down)
        ctx["lambda_2"] = {"type": "uniform", "low": lo, "high": hi}
        for k in (1, 3):
            ctx[f"_f{k}"] = {"type": "uniform", "low": 0.0, "high": 1.0}
            ctx[f"_u{k}"] = {"type": "uniform", "low": 0.0, "high": 1.0}
            ctx[f"_up{k}"] = (f"np.logical_and(2 * lambda_2 < {hi}, "
                              f"np.logical_or(_f{k} < 0.5, lambda_2 / 2 <= {lo}))")
            ctx[f"lambda_{k}"] = (f"np.where(_up{k}, 2 * lambda_2 + ({hi} - 2 * lambda_2) * _u{k}, "
                                  f"{lo} + (lambda_2 / 2 - {lo}) * _u{k})")
    # death rates: drawn independently in both sets
    for k in (1, 2, 3):
        ctx[f"mu_{k}"] = {"type": "uniform", "low": MU_RANGE[0], "high": MU_RANGE[1]}

    def rates(values):
        return {"value": values, "change_times": ["change_time_1", "change_time_2"]}

    return {
        "output_dir": output_dir,
        "data_type": "tree",
        "n_samples": n_samples,
        "n_jobs": N_JOBS,
        "seed": SEED,
        "context": ctx,
        "tree_generator": {
            "spec": "phylogenie.canonical",
            "states": ["I"],
            "init_state": "I",
            "n_leaves": N_LEAVES,
            "max_time": MAX_TIME,
            "timeout": TIMEOUT,
            "birth_rates": rates(["lambda_1", "lambda_2", "lambda_3"]),
            "death_rates": rates(["mu_1", "mu_2", "mu_3"]),
            "sampling_rates": rates([PSI, PSI, PSI]),
            "remove_after_sampling": False,   # sampled individuals stay in the tree (sampled ancestors)
        },
    }


def run_name(set_name, n):
    return f"3-Interval-ChangeTime-{set_name}-{n}-{args.tag}-seed{SEED}"


def end_time(path):
    """Largest distance from the origin to any node of the tree."""
    tree = load_newick(path)[0]
    best = 0.0
    stack = [(tree, tree.branch_length or 0.0)]
    while stack:
        node, d = stack.pop()
        best = max(best, d)
        stack.extend((c, d + (c.branch_length or 0.0)) for c in node.children)
    return best


# ---------------- main
def main():
    exe = shutil.which("phylogenie")
    if exe is None:
        sys.exit("The phylogenie command was not found. Install it with: pip install -r requirements.txt")

    for s in SETS:
        name = run_name(s, N_TREES[s])

        # 1. write the Phylogenie settings and simulate
        with open(f"{name}.yaml", "w") as f:
            yaml.dump(make_config(s, f"outputs/{name}", N_TREES[s]), f, sort_keys=False)
        shutil.rmtree(f"outputs/{name}", ignore_errors=True)
        print(f"[{s}] simulating {N_TREES[s]} trees, {N_LEAVES} tips, 3 intervals, seed {SEED} ...", flush=True)
        t0 = time.time()
        if subprocess.run([exe, f"{name}.yaml"]).returncode != 0:
            sys.exit(f"phylogenie failed for set {s}")
        minutes = (time.time() - t0) / 60
        print(f"[{s}] simulation done in {minutes:.1f} minutes", flush=True)

        # 2. filter: keep trees with end time >= change_time_2 + 1
        out_dir = f"outputs/{name}"
        m = pd.read_csv(f"{out_dir}/metadata.csv")
        m["end_time"] = [end_time(f"{out_dir}/data/{i}.nwk") for i in m["file_id"]]
        m["keep"] = m["end_time"] >= m["change_time_2"] + SURVIVE_AFTER_CHANGE
        kept_dir = f"outputs/{name}-kept"
        shutil.rmtree(kept_dir, ignore_errors=True)
        os.makedirs(f"{kept_dir}/data")
        k = m[m["keep"]].copy()
        for i in k["file_id"]:
            shutil.copy(f"{out_dir}/data/{i}.nwk", f"{kept_dir}/data/{i}.nwk")
        print(f"[{s}] {len(m)} simulated, {len(k)} kept ({100 * len(k) / len(m):.1f}%)", flush=True)

        # 3. save the true values and a short summary
        k = k[["file_id"] + TARGETS + ["end_time"]].reset_index(drop=True)
        k.to_csv(f"{kept_dir}/metadata.csv", index=False)
        np.save(f"{kept_dir}/y.npy", k[TARGETS].to_numpy(dtype=np.float32))
        np.save(f"{kept_dir}/file_ids.npy", k["file_id"].to_numpy())
        with open(f"{kept_dir}/y_columns.txt", "w") as f:
            f.write("\n".join(TARGETS) + "\n")
        shutil.copy(f"{name}.yaml", kept_dir)
        with open(f"{kept_dir}/run_info.txt", "w") as f:
            f.write(f"set: {s}\nintervals: 3\ntips: {N_LEAVES}\nseed: {SEED}\ntimeout: {TIMEOUT}\n"
                    f"lambda: {LAMBDA_RANGE}, mu: {MU_RANGE}, psi: {PSI}\n"
                    f"change times: 2 random in {CHANGE_RANGE}, gap >= {args.min_gap}\n"
                    f"run time (minutes): {minutes:.1f}\nsimulated: {len(m)}\n"
                    f"kept (end time >= change_time_2 + {SURVIVE_AFTER_CHANGE}): {len(k)}\n")

        # 4. zip the kept folder
        shutil.make_archive(f"{name}-kept", "zip", root_dir=".", base_dir=kept_dir)
        print(f"[{s}] {len(k)} trees x {len(TARGETS)} values saved -> {name}-kept.zip", flush=True)

    print("Done. The kept trees are in outputs/ and in the -kept.zip files.")


if __name__ == "__main__":
    main()
