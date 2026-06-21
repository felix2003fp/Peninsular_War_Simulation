import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ownership codes (match Env/config.py; ownership stores these directly)
FRANCE, ALLIES, NEUTRAL = 0, 1, -1

TIER_COLORS = {1: "#1f77b4", 2: "#d62728"}


# Loading

def load_truth(path):
    df = pd.read_csv(path)
    df = df.set_index("node_id")
    week_cols = [c for c in df.columns if c.startswith("week_")]
    week_cols.sort(key=lambda c: int(c.split("_")[1]))
    truth = df[week_cols].astype(int)            # rows = cities, cols = weeks, 1=French
    return truth                                  # DataFrame (city x week)


def discover_tiers(results_dir):
    tiers = {}
    for p in sorted(results_dir.glob("ownership_tier*.npz")):
        t = int(p.stem.replace("ownership_tier", ""))
        ep = results_dir / f"episodes_tier{t}.csv"
        tiers[t] = {
            "npz": p,
            "episodes": pd.read_csv(ep) if ep.exists() else None,
        }
    return tiers


def load_tier(info):
    z = np.load(info["npz"], allow_pickle=False)
    own = z["ownership"]                          # (n_sims, n_weeks, n_nodes)
    nodes = [str(x) for x in z["node_ids"]]
    seeds = z["seeds"] if "seeds" in z.files else np.arange(own.shape[0])
    return own, nodes, seeds


# Metrics

def p_france(own, node_idx):
    """
    P(France holds) per week for one node: mean over sims of (owner==FRANCE).
    """
    return (own[:, :, node_idx] == FRANCE).mean(axis=0)


def first_french_week(series_2d):
    """
    Per sim, first week the node is French (NaN if never). series_2d: (n_sims, n_weeks).
    """
    is_fr = series_2d == FRANCE
    has = is_fr.any(axis=1)
    first = np.argmax(is_fr, axis=1).astype(float)
    first[~has] = np.nan
    return first


def real_first_week(truth_row):
    idx = np.where(truth_row.values == 1)[0]
    return int(idx[0]) if len(idx) else None


def spearman(a, b):
    """
    Rank correlation without scipy. Pairs with NaN are dropped.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 2:
        return np.nan
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


# Figures

def fig_prob_curves(truth, tiers_data, weeks, out):
    cities = list(truth.index)
    ncol = 2
    nrow = int(np.ceil(len(cities) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.2 * nrow),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    xmax = len(weeks) - 1
    for ax, city in zip(axes, cities):
        ax.step(weeks, truth.loc[city].values, where="post",
                color="black", lw=1.6, label="real")
        for t, d in tiers_data.items():
            if city in d["node_idx"]:
                ax.plot(weeks, d["pf"][city], color=TIER_COLORS.get(t, None),
                        lw=1.3, alpha=0.85, label=f"Tier {t}")
        ax.set_title(city, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, xmax)
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
        ax.tick_params(labelbottom=True)   # x-axis labels on every panel
    for ax in axes[len(cities):]:
        ax.axis("off")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("P(France holds) — ensemble vs real war   (1=French, 0=Allied)", y=1.0)
    fig.text(0.5, -0.01, "week (0 = Nov 1808)", ha="center")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_fidelity(truth, tiers_data, weeks, out):
    fig, ax = plt.subplots(figsize=(10, 4))
    truth_mat = truth.values  # (cities, weeks), 1=French
    for t, d in tiers_data.items():
        cities = [c for c in truth.index if c in d["node_idx"]]
        modal = np.vstack([(d["pf"][c] >= 0.5).astype(int) for c in cities])  # (cities, weeks)
        tr = truth.loc[cities].values
        fidelity = (modal == tr).mean(axis=0)     # per week, share of cities matching
        ax.plot(weeks, fidelity, color=TIER_COLORS.get(t, None), lw=1.6,
                label=f"Tier {t}  (mean {fidelity.mean():.2f})")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("week (0 = Nov 1808)")
    ax.set_ylabel("share of key cities matching history")
    ax.set_title("Historical fidelity over time (majority-vote ownership vs real)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_time_to_capture(truth, tiers_data, out):
    cities = list(truth.index)
    fig, ax = plt.subplots(figsize=(11, 5))
    n_t = len(tiers_data)
    width = 0.8 / max(n_t, 1)
    positions = np.arange(len(cities))
    for k, (t, d) in enumerate(sorted(tiers_data.items())):
        data = []
        for c in cities:
            if c in d["node_idx"]:
                ff = first_french_week(d["own"][:, :, d["node_idx"][c]])
                data.append(ff[~np.isnan(ff)])
            else:
                data.append(np.array([]))
        pos = positions + (k - (n_t - 1) / 2) * width
        bp = ax.boxplot(
            [x if len(x) else [np.nan] for x in data],
            positions=pos, widths=width * 0.9, patch_artist=True,
            showfliers=False, manage_ticks=False,
        )
        for box in bp["boxes"]:
            box.set(facecolor=TIER_COLORS.get(t, "grey"), alpha=0.5)
        for med in bp["medians"]:
            med.set(color="black")

    # Real first-French weeks as markers
    for i, c in enumerate(cities):
        rw = real_first_week(truth.loc[c])
        if rw is not None:
            ax.scatter(i, rw, marker="*", s=130, color="gold",
                       edgecolor="black", zorder=5,
                       label="real" if i == 0 else None)
    ax.set_xticks(positions)
    ax.set_xticklabels(cities)
    ax.set_ylabel("first week under French control")
    ax.set_title("Time-to-first-French-capture: simulated distribution vs real (★)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_outcomes(tiers_data, out):
    have_ep = {t: d for t, d in tiers_data.items() if d["episodes"] is not None}
    if not have_ep:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # winner distribution
    ax = axes[0]
    cats = ["France", "Allies", "Draw"]
    x = np.arange(len(cats)); n_t = len(have_ep); w = 0.8 / n_t
    for k, (t, d) in enumerate(sorted(have_ep.items())):
        vc = d["episodes"]["winner"].value_counts()
        frac = [vc.get(c, 0) / len(d["episodes"]) for c in cats]
        ax.bar(x + (k - (n_t - 1) / 2) * w, frac, w, color=TIER_COLORS.get(t),
               label=f"Tier {t}")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("share of simulations"); ax.set_title("War winner")
    ax.axhline(0, color="k", lw=0.5); ax.legend()

    # End reason
    ax = axes[1]
    reasons = sorted(set().union(*[set(d["episodes"]["end_reason"].unique()) for d in have_ep.values()]))
    x = np.arange(len(reasons))
    for k, (t, d) in enumerate(sorted(have_ep.items())):
        vc = d["episodes"]["end_reason"].value_counts()
        frac = [vc.get(r, 0) / len(d["episodes"]) for r in reasons]
        ax.bar(x + (k - (n_t - 1) / 2) * w, frac, w, color=TIER_COLORS.get(t))
    ax.set_xticks(x); ax.set_xticklabels(reasons, rotation=30, ha="right")
    ax.set_ylabel("share"); ax.set_title("How the war ended")

    # End turn histogram
    ax = axes[2]
    for t, d in sorted(have_ep.items()):
        ax.hist(d["episodes"]["end_turn"], bins=30, alpha=0.5,
                color=TIER_COLORS.get(t), label=f"Tier {t}")
    ax.set_xlabel("end turn (week)"); ax.set_ylabel("count"); ax.set_title("Game length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


# Representative episodes

def sim_distance_to_history(own, node_idx, truth):
    """Per-sim mean |is_French_sim - truth_French| over key cities x weeks (lower = closer)."""
    cities = [c for c in truth.index if c in node_idx]
    idxs = [node_idx[c] for c in cities]
    is_fr = (own[:, :, idxs] == FRANCE).astype(float)     # (n_sims, weeks, cities)
    tr = truth.loc[cities].values.T[None, :, :]           # (1, weeks, cities)
    return np.abs(is_fr - tr).mean(axis=(1, 2))           # (n_sims,)


def representative(seeds, dist, episodes):
    order = np.argsort(dist)
    closest = int(order[0])
    median = int(order[len(order) // 2])
    out = {
        "closest_to_history": {"sim": closest, "seed": int(seeds[closest]), "distance": float(dist[closest])},
        "median": {"sim": median, "seed": int(seeds[median]), "distance": float(dist[median])},
    }
    if episodes is not None and "winner" in episodes:
        fr_wins = np.where(episodes["winner"].values == "France")[0]
        if len(fr_wins):
            best = fr_wins[np.argmin(dist[fr_wins])]
            out["typical_french_win"] = {"sim": int(best), "seed": int(seeds[best]),
                                          "distance": float(dist[best])}
    return out


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--truth", default="historical_truth.csv")
    ap.add_argument("--out", default=None, help="output dir (default <results>/analysis)")
    args = ap.parse_args()

    root = Path(__file__).parent
    results_dir = (root / args.results) if not Path(args.results).is_absolute() else Path(args.results)
    truth_path = (root / args.truth) if not Path(args.truth).is_absolute() else Path(args.truth)
    out_dir = Path(args.out) if args.out else results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = load_truth(truth_path)
    n_weeks = truth.shape[1]
    weeks = np.arange(n_weeks)

    tiers = discover_tiers(results_dir)
    if not tiers:
        raise SystemExit(f"No ownership_tier*.npz found in {results_dir}")

    tiers_data = {}
    for t, info in tiers.items():
        own, nodes, seeds = load_tier(info)
        if own.shape[1] != n_weeks:
            raise SystemExit(f"Tier {t} has {own.shape[1]} weeks but truth has {n_weeks}")
        node_idx = {c: nodes.index(c) for c in truth.index if c in nodes}
        missing = [c for c in truth.index if c not in node_idx]
        if missing:
            print(f"  [warn] Tier {t}: cities absent from node list: {missing}")
        pf = {c: p_france(own, node_idx[c]) for c in node_idx}
        tiers_data[t] = {"own": own, "seeds": seeds, "node_idx": node_idx,
                         "pf": pf, "episodes": info["episodes"]}

    # Per-city metrics table
    rows = []
    for t, d in tiers_data.items():
        for c in truth.index:
            if c not in d["node_idx"]:
                continue
            tr = truth.loc[c].values.astype(float)
            pf = d["pf"][c]
            brier = float(np.mean((pf - tr) ** 2))
            agreement = float(np.mean((pf >= 0.5).astype(int) == tr.astype(int)))
            ff = first_french_week(d["own"][:, :, d["node_idx"][c]])
            rows.append({
                "city": c, "tier": t,
                "brier": round(brier, 4),
                "agreement": round(agreement, 4),
                "real_first_french_week": real_first_week(truth.loc[c]),
                "sim_median_first_french_week":
                    (None if np.all(np.isnan(ff)) else float(np.nanmedian(ff))),
                "sim_ever_french_rate": round(float((~np.isnan(ff)).mean()), 4),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "city_metrics.csv", index=False)

    # Figures
    fig_prob_curves(truth, tiers_data, weeks, out_dir / "prob_curves.png")
    fig_fidelity(truth, tiers_data, weeks, out_dir / "fidelity_over_time.png")
    fig_time_to_capture(truth, tiers_data, out_dir / "time_to_capture.png")
    has_outcomes = fig_outcomes(tiers_data, out_dir / "outcomes.png")

    # Capture-order correlation + representative episodes + report
    lines = ["# Peninsular War — simulation vs reality\n"]
    lines.append(f"Key cities: {', '.join(truth.index)}\n")
    for t, d in sorted(tiers_data.items()):
        sub = metrics[metrics.tier == t]
        lines.append(f"\n## Tier {t}\n")
        lines.append(f"- Mean Brier (lower=better): **{sub['brier'].mean():.3f}**")
        lines.append(f"- Mean weekly agreement (majority vote vs history): **{sub['agreement'].mean():.3f}**")

        # Capture order
        real = [real_first_week(truth.loc[c]) for c in truth.index if c in d["node_idx"]]
        simm = [sub.set_index("city").loc[c, "sim_median_first_french_week"]
                for c in truth.index if c in d["node_idx"]]
        real = [np.nan if r is None else r for r in real]
        rho = spearman(real, simm)
        lines.append(f"- Capture-order rank correlation (real vs sim median): **{rho:.2f}**")
        if d["episodes"] is not None:
            ep = d["episodes"]
            wr = (ep["winner"] == "France").mean()
            lines.append(f"- French win rate: **{wr:.1%}**  (real outcome: Allied victory)")
            lines.append(f"- Mean game length: {ep['end_turn'].mean():.0f} weeks; "
                         f"ended by: {dict(ep['end_reason'].value_counts())}")
        dist = sim_distance_to_history(d["own"], d["node_idx"], truth)
        rep = representative(d["seeds"], dist, d["episodes"])
        lines.append("- Representative episodes (replay seed with simulate_history "
                     "`--base-seed <seed> --n-sims 1`):")
        for name, r in rep.items():
            lines.append(f"    - {name}: seed **{r['seed']}** (distance {r['distance']:.3f})")
    lines.append("\n## Files\n")
    lines.append("- `prob_curves.png` — per-city P(France) vs real step function")
    lines.append("- `fidelity_over_time.png` — share of cities matching history each week")
    lines.append("- `time_to_capture.png` — simulated first-capture spread vs real date (★)")
    if has_outcomes:
        lines.append("- `outcomes.png` — winner / end-reason / game-length distributions")
    lines.append("- `city_metrics.csv` — per-city Brier, agreement, capture weeks")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote analysis to {out_dir}")
    print("\n".join(l for l in lines if l.startswith(("##", "- Mean", "- French", "- Capture"))))


if __name__ == "__main__":
    main()
