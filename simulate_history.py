import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from Env.config import FRANCE, ALLIES, NEUTRAL, MAX_ARMIES, MAX_TURNS

# 313 weekly columns: week 0 = reset/anchor state, weeks 1..312 = after each step
N_WEEKS = MAX_TURNS  # 313

# Default model locations (phase-3 France, phase-4 Allies)
TIER_MODELS = {
    1: {
        "france": "models/Tier-1/20260611-233608-t1-phase3-france/final.zip",
        "allies": "models/Tier-1/20260611-233608-t1-phase4-allies/final.zip",
    },
    2: {
        "france": "models/Tier-2/20260613-000313-t2-phase3-france/final.zip",
        "allies": "models/Tier-2/20260613-000313-t2-phase4-allies/final.zip",
    },
}

# Per-worker globals
_FR_MODEL = None
_AL_MODEL = None
_ENV = None


def _worker_init(france_path, allies_path):
    """
    Load both models and build one env per worker process (once).
    """
    global _FR_MODEL, _AL_MODEL, _ENV
    # Keep each process single-threaded.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    from sb3_contrib import MaskablePPO
    from Env import PeninsularWarEnv
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    no_buffer = {"n_steps": 1}
    _FR_MODEL = MaskablePPO.load(france_path, device="cpu", custom_objects=no_buffer)
    _AL_MODEL = MaskablePPO.load(allies_path, device="cpu", custom_objects=no_buffer)
    _ENV = PeninsularWarEnv(render_mode=None, verbose=False)


def _model_action(model, env, faction, deterministic=False):
    """
    One faction's full MAX_ARMIES action array (mirrors watch.py.model_action).

    Auto-detects Tier-1 (joint MultiDiscrete) vs Tier-2 (per-army Discrete).
    """
    from gymnasium.spaces import Discrete
    from Env.sb3_wrapper import (
        _flat_action_mask, _build_local_obs, N_ACTIONS_PER_ARMY,
    )

    # Tier 2 — decide one army at a time from its 190-dim local obs
    if isinstance(model.action_space, Discrete):
        active = env._active_army_slots(faction)
        base_obs = env._get_obs(faction)
        legal = env.legal_actions(faction)
        action = np.zeros(MAX_ARMIES, dtype=np.int64)
        for slot in range(len(active)):
            obs = _build_local_obs(env, faction, base_obs, slot)
            mask = np.zeros(N_ACTIONS_PER_ARMY, dtype=bool)
            if slot in legal:
                for d in legal[slot]:
                    mask[d] = True
            else:
                mask[0] = True
            a, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            action[slot] = int(a)
        return action

    # Tier 1 — joint action over the full faction observation
    obs = env._get_obs(faction)
    mask = _flat_action_mask(env, faction)
    action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
    return np.asarray(action, dtype=np.int64)


def _classify_end(info, terminated, truncated):
    """
    Return (winner, end_reason) from the final info dict.
    """
    if terminated:
        dom = info.get("dominance_winner", NEUTRAL)
        if dom == FRANCE:
            return "France", "france_dominance"
        if dom == ALLIES:
            return "Allies", "allies_dominance"
        # Otherwise a side was wiped out
        if info.get("allies_troops_total", 1) == 0:
            return "France", "allies_eliminated"
        if info.get("france_troops_total", 1) == 0:
            return "Allies", "france_eliminated"
        return "Draw", "terminated_unknown"
    # Truncation at the turn limit -> decide on node count
    f_n, a_n = info["france_nodes"], info["allies_nodes"]
    winner = "France" if f_n > a_n else ("Allies" if a_n > f_n else "Draw")
    return winner, "turn_limit"


def run_episode(args):
    """
    Play one full game; return (idx, ownership[N_WEEKS, N] int8, summary).
    """
    idx, seed = args
    env = _ENV
    N = env.N

    # Seed everything for reproducibility; diversity comes from deterministic=False
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

    _, info = env.reset(seed=seed)

    own = np.empty((N_WEEKS, N), dtype=np.int8)
    own[0] = env.owner                       # week 0 = anchor, reset state

    terminated = truncated = False
    last_info = info
    for week in range(1, N_WEEKS):           # weeks 1..312 (312 steps)
        f_action = _model_action(_FR_MODEL, env, FRANCE, deterministic=False)
        a_action = _model_action(_AL_MODEL, env, ALLIES, deterministic=False)
        _, _, terminated, truncated, info = env.step(
            {"france": f_action, "allies": a_action}
        )
        own[week] = env.owner
        last_info = info
        if terminated or truncated:
            if week + 1 < N_WEEKS:           # forward-fill the rest of the grid
                own[week + 1:] = env.owner
            break

    winner, end_reason = _classify_end(last_info, terminated, truncated)
    summary = {
        "sim_id": idx,
        "seed": seed,
        "winner": winner,
        "end_reason": end_reason,
        "end_turn": int(last_info["turn"]),
        "france_nodes": int(last_info["france_nodes"]),
        "allies_nodes": int(last_info["allies_nodes"]),
        "neutral_nodes": int(last_info.get("neutral_nodes", 0)),
        "france_troops_total": int(last_info.get("france_troops_total", 0)),
        "allies_troops_total": int(last_info.get("allies_troops_total", 0)),
    }
    return idx, own, summary


def _git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def simulate_tier(tier, n_sims, base_seed, workers, out_dir):
    fr_path = str((ROOT / TIER_MODELS[tier]["france"]).resolve())
    al_path = str((ROOT / TIER_MODELS[tier]["allies"]).resolve())
    for label, p in (("France", fr_path), ("Allies", al_path)):
        if not Path(p).exists():
            sys.exit(f"  [sim] Tier {tier} {label} model not found: {p}")

    # Node order (same for every run)
    from Env import PeninsularWarEnv
    node_ids = list(PeninsularWarEnv(render_mode=None, verbose=False).node_ids)
    N = len(node_ids)

    seeds = [base_seed + i for i in range(n_sims)]
    tasks = list(enumerate(seeds))

    ownership = np.empty((n_sims, N_WEEKS, N), dtype=np.int8)
    summaries = [None] * n_sims

    print(f"\n  ── Tier {tier} ──  {n_sims} sims, {workers} worker(s)")
    print(f"     France : {Path(fr_path).parent.name}")
    print(f"     Allies : {Path(al_path).parent.name}")
    print(f"     loading models in each worker ...", flush=True)
    t0 = time.time()
    done = 0

    def _should_print(d):
        # Print for the first 5 to see if it's working, then occasionally
        return d <= 5 or d % 25 == 0 or d == n_sims

    if workers <= 1:
        _worker_init(fr_path, al_path)
        for task in tasks:
            idx, own, summ = run_episode(task)
            ownership[idx] = own
            summaries[idx] = summ
            done += 1
            if _should_print(done):
                _progress(done, n_sims, t0)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init,
            initargs=(fr_path, al_path),
        ) as ex:
            for idx, own, summ in ex.map(run_episode, tasks, chunksize=1):
                ownership[idx] = own
                summaries[idx] = summ
                done += 1
                if _should_print(done):
                    _progress(done, n_sims, t0)

    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"ownership_tier{tier}.npz"
    np.savez_compressed(
        npz_path,
        ownership=ownership,
        node_ids=np.array(node_ids),
        seeds=np.array(seeds),
        codes=np.array([f"FRANCE={FRANCE}", f"ALLIES={ALLIES}", f"NEUTRAL={NEUTRAL}"]),
    )

    csv_path = out_dir / f"episodes_tier{tier}.csv"
    _write_csv(csv_path, summaries)

    meta = {
        "tier": tier,
        "n_sims": n_sims,
        "n_weeks": N_WEEKS,
        "n_nodes": N,
        "base_seed": base_seed,
        "deterministic": False,
        "anchor_week0": "1808-11-04",
        "codes": {"FRANCE": FRANCE, "ALLIES": ALLIES, "NEUTRAL": NEUTRAL},
        "france_model": fr_path,
        "allies_model": al_path,
        "git_rev": _git_rev(),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    meta_path = out_dir / f"metadata_tier{tier}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # Quick win-rate readout
    wins = {}
    for s in summaries:
        wins[s["winner"]] = wins.get(s["winner"], 0) + 1
    print(f"     saved -> {npz_path.name}, {csv_path.name}, {meta_path.name}")
    print(f"     outcomes: {wins}   ({meta['elapsed_sec']}s)")
    return ownership, summaries


def _progress(done, total, t0):
    el = time.time() - t0
    rate = done / el if el > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(f"     {done}/{total}  ({rate:.2f}/s, ETA {eta/60:.1f} min)", flush=True)


def _write_csv(path, summaries):
    import csv
    cols = ["sim_id", "seed", "winner", "end_reason", "end_turn",
            "france_nodes", "allies_nodes", "neutral_nodes",
            "france_troops_total", "allies_troops_total"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for s in summaries:
            wr.writerow(s)


def main():
    ap = argparse.ArgumentParser(description="Run self-play ensembles and record node ownership.")
    ap.add_argument("--tiers", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--base-seed", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    for tier in args.tiers:
        simulate_tier(tier, args.n_sims, args.base_seed, args.workers, out_dir)


if __name__ == "__main__":
    main()
