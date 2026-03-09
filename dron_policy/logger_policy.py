import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from example_usage import CooperativeReaching


# =============================================================================
# PROBE CONSTANTS
# =============================================================================

PROBE_SEEDS      = [11, 17, 23]
PROBE_ACTION_SEQ = [0, 0, 2, 0, 2]


# =============================================================================
# PROBE CONSTRUCTION
# =============================================================================

def make_probe(opponent_type, seed):
    np.random.seed(seed)
    env = CooperativeReaching(opponent_type=opponent_type)
    state, opp = env.reset()
    done = False
    for a in PROBE_ACTION_SEQ:
        if done:
            break
        state, opp, _, done = env.step(a)
    return state, opp


def build_probes(opponent_types):
    probes = {}
    for t in opponent_types:
        probes[t] = []
        for seed in PROBE_SEEDS:
            probes[t].append(make_probe(t, seed))
    return probes


# =============================================================================
# CSV HELPERS
# =============================================================================

def append_row(csv_path, fieldnames, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def entropy_of_probs(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    return float(-(p * np.log(p)).sum())


def log_dron_policy_probe(csv_path, episode, split, opponent, probe_id,
                           probs, value, gating_w):
    A = probs.shape[0]
    K = gating_w.shape[0]

    row = {
        "episode":       episode,
        "split":         split,
        "opponent":      opponent,
        "probe_id":      probe_id,
        "chosen_action": int(np.argmax(probs)),
        "entropy":       entropy_of_probs(probs),
        "value":         float(value),
    }
    for a in range(A):
        row[f"p_a{a}"] = float(probs[a])
    for k in range(K):
        row[f"w_k{k}"] = float(gating_w[k])

    append_row(csv_path, list(row.keys()), row)


def log_baseline_policy_probe(csv_path, episode, split, opponent, probe_id,
                               probs, value):
    A = probs.shape[0]

    row = {
        "episode":       episode,
        "split":         split,
        "opponent":      opponent,
        "probe_id":      probe_id,
        "chosen_action": int(np.argmax(probs)),
        "entropy":       entropy_of_probs(probs),
        "value":         float(value),
    }
    for a in range(A):
        row[f"p_a{a}"] = float(probs[a])

    append_row(csv_path, list(row.keys()), row)


# =============================================================================
# PROBE EVALUATION
# =============================================================================

def evaluate_probes(dron, baseline, probes_seen, probes_unseen, ep,
                    csv_dron="probe_policy_dron.csv",
                    csv_base="probe_policy_baseline.csv"):
    for split_name, probe_dict in [("seen", probes_seen), ("unseen", probes_unseen)]:
        for opp_type, probe_list in probe_dict.items():
            for pid, (s, o) in enumerate(probe_list):
                p, v, w = dron.inspect(s, o, include_experts=False)
                log_dron_policy_probe(csv_dron, ep, split_name, opp_type, pid, p, v, w)

                pb, vb = baseline.inspect(s, o)
                log_baseline_policy_probe(csv_base, ep, split_name, opp_type, pid, pb, vb)


# =============================================================================
# PLOTTING
# =============================================================================

def smooth_xy(x, y, window=10):
    x = np.asarray(x)
    y = np.asarray(y, dtype=np.float32)
    if len(y) < window:
        return x, y
    y_s = np.convolve(y, np.ones(window) / window, mode="valid")
    x_s = x[window - 1:]
    return x_s, y_s


def plot_probe_trends(csv_dron="probe_policy_dron.csv"):
    rows = []
    with open(csv_dron, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("No probe CSV rows found; skipping probe plots.")
        return

    episodes = sorted({int(r["episode"]) for r in rows})
    splits   = ["seen", "unseen"]
    w_cols   = sorted([k for k in rows[0].keys() if k.startswith("w_k")])

    agg = {
        (ep, sp): {"entropy": [], **{c: [] for c in w_cols}}
        for ep in episodes for sp in splits
    }

    for r in rows:
        ep = int(r["episode"])
        sp = r["split"]
        if sp not in splits:
            continue
        agg[(ep, sp)]["entropy"].append(float(r["entropy"]))
        for c in w_cols:
            agg[(ep, sp)][c].append(float(r[c]))

    def mean_or_nan(x):
        return float(np.mean(x)) if len(x) else np.nan

    series = {
        sp: {"episode": [], "entropy": [], **{c: [] for c in w_cols}}
        for sp in splits
    }
    for sp in splits:
        for ep in episodes:
            series[sp]["episode"].append(ep)
            series[sp]["entropy"].append(mean_or_nan(agg[(ep, sp)]["entropy"]))
            for c in w_cols:
                series[sp][c].append(mean_or_nan(agg[(ep, sp)][c]))

    # Entropy plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for sp in splits:
        x, y = smooth_xy(series[sp]["episode"], series[sp]["entropy"], window=10)
        ax.plot(x, y, label=f"entropy ({sp})", linewidth=2)
    ax.set_title("DRON-MoE policy probes: action-prob entropy over training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("probe_policy_trends_entropy.png", dpi=300)
    print("Saved: probe_policy_trends_entropy.png")
    plt.show()

    # Gating weights plot
    if w_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        for sp in splits:
            for c in w_cols:
                x, y = smooth_xy(series[sp]["episode"], series[sp][c], window=10)
                ax.plot(x, y, label=f"{c} ({sp})", alpha=0.85)
        ax.set_title("DRON-MoE policy probes: gating weights over training")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.3)
        ax.legend(ncols=2, fontsize=9)
        plt.tight_layout()
        plt.savefig("probe_policy_trends_gating.png", dpi=300)
        print("Saved: probe_policy_trends_gating.png")
        plt.show()
