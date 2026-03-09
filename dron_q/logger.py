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


def log_dron_probe(csv_path, episode, split, opponent, probe_id, q_final, q_experts, w):
    A = q_final.shape[0]
    K = w.shape[0]

    row = {
        "episode":       episode,
        "split":         split,
        "opponent":      opponent,
        "probe_id":      probe_id,
        "chosen_action": int(np.argmax(q_final)),
        "q_max":         float(np.max(q_final)),
        "q_gap_top2":    float(np.sort(q_final)[-1] - np.sort(q_final)[-2]),
    }
    for a in range(A):
        row[f"q_final_a{a}"] = float(q_final[a])
    for k in range(K):
        row[f"w_k{k}"] = float(w[k])
    for k in range(K):
        for a in range(A):
            row[f"q_expert{k}_a{a}"] = float(q_experts[a, k])

    append_row(csv_path, list(row.keys()), row)


def log_baseline_probe(csv_path, episode, split, opponent, probe_id, q):
    A = q.shape[0]
    row = {
        "episode":       episode,
        "split":         split,
        "opponent":      opponent,
        "probe_id":      probe_id,
        "chosen_action": int(np.argmax(q)),
        "q_max":         float(np.max(q)),
        "q_gap_top2":    float(np.sort(q)[-1] - np.sort(q)[-2]),
    }
    for a in range(A):
        row[f"q_a{a}"] = float(q[a])

    append_row(csv_path, list(row.keys()), row)


# =============================================================================
# PROBE EVALUATION
# =============================================================================

def evaluate_probes(dron, baseline, probes_seen, probes_unseen, ep,
                    csv_dron="probe_qvalues_dron.csv",
                    csv_base="probe_qvalues_baseline.csv"):
    for split_name, probe_dict in [("seen", probes_seen), ("unseen", probes_unseen)]:
        for opp_type, probe_list in probe_dict.items():
            for pid, (s, o) in enumerate(probe_list):
                qf, qe, w = dron.inspect(s, o)
                log_dron_probe(csv_dron, ep, split_name, opp_type, pid, qf, qe, w)

                qb = baseline.inspect(s)
                log_baseline_probe(csv_base, ep, split_name, opp_type, pid, qb)


# =============================================================================
# PLOTTING
# =============================================================================

def smooth_xy(x, y, window=10):
    x = np.asarray(x)
    y = np.asarray(y, dtype=np.float32)
    # ✅ Fall back to available data size instead of returning empty
    if len(y) < window:
        window = max(1, len(y))
    y_s = np.convolve(y, np.ones(window) / window, mode="valid")
    x_s = x[window - 1:]
    return x_s, y_s


def plot_probe_trends(csv_dron="probe_qvalues_dron.csv"):
    rows = []
    with open(csv_dron, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("No probe CSV rows found; skipping plot.")
        return

    episodes = sorted({int(r["episode"]) for r in rows})
    splits   = ["seen", "unseen"]
    w_cols   = sorted([k for k in rows[0].keys() if k.startswith("w_k")])

    agg = {
        (ep, sp): {"q_max": [], "q_gap_top2": [], **{c: [] for c in w_cols}}
        for ep in episodes for sp in splits
    }

    for r in rows:
        ep = int(r["episode"])
        sp = r["split"]
        if sp not in splits:
            continue
        agg[(ep, sp)]["q_max"].append(float(r["q_max"]))
        agg[(ep, sp)]["q_gap_top2"].append(float(r["q_gap_top2"]))
        for c in w_cols:
            agg[(ep, sp)][c].append(float(r[c]))

    def mean_or_nan(x):
        return float(np.mean(x)) if len(x) else np.nan

    series = {
        sp: {"episode": [], "q_max": [], "q_gap_top2": [], **{c: [] for c in w_cols}}
        for sp in splits
    }
    for sp in splits:
        for ep in episodes:
            series[sp]["episode"].append(ep)
            series[sp]["q_max"].append(mean_or_nan(agg[(ep, sp)]["q_max"]))
            series[sp]["q_gap_top2"].append(mean_or_nan(agg[(ep, sp)]["q_gap_top2"]))
            for c in w_cols:
                series[sp][c].append(mean_or_nan(agg[(ep, sp)][c]))

    # Q-max and Q-gap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for sp in splits:
        x, y = smooth_xy(np.array(series[sp]["episode"]), series[sp]["q_max"])
        ax1.plot(x, y, label=f"q_max ({sp})", linewidth=2)
        x, y = smooth_xy(np.array(series[sp]["episode"]), series[sp]["q_gap_top2"])
        ax2.plot(x, y, label=f"q_gap_top2 ({sp})", linewidth=2)

    ax1.set_title("DRON-MoE probe: max Q over training")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Max Q")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title("DRON-MoE probe: top-2 Q gap over training")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Q(best) - Q(2nd)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("probe_trends_qmax_qgap.png", dpi=300)
    print("Saved: probe_trends_qmax_qgap.png")
    plt.show()

    # Gating weights
    fig, ax = plt.subplots(figsize=(10, 6))
    for sp in splits:
        for c in w_cols:
            x, y = smooth_xy(np.array(series[sp]["episode"]), series[sp][c])
            ax.plot(x, y, label=f"{c} ({sp})", alpha=0.85, linewidth=2)

    ax.set_title("DRON-MoE probe: gating weights over training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=2, fontsize=9)

    plt.tight_layout()
    plt.savefig("probe_trends_gating_weights.png", dpi=300)
    print("Saved: probe_trends_gating_weights.png")
    plt.show()
