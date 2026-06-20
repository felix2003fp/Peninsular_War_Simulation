import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from Env import PeninsularWarEnv
from Env.renderer import MapRenderer
from Env.config import FRANCE, ALLIES, MAX_ARMIES


# ── Default opponent model per tier (phase-3 France, phase-4 Allies) ───────────
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


# ── Agents ────────────────────────────────────────────────────────────────────

def random_action(env: PeninsularWarEnv, faction: int) -> np.ndarray:
    """Pick a uniformly random legal direction for every army of the faction."""
    legal  = env.legal_actions(faction)
    action = np.zeros(MAX_ARMIES, dtype=np.int64)
    for slot, dirs in legal.items():
        action[slot] = np.random.choice(dirs)
    return action


def model_action(model, env, faction, deterministic=True):
    """One faction's full MAX_ARMIES action array from a trained MaskablePPO model.

    Auto-detects Tier 1 (joint MultiDiscrete over the full obs) vs Tier 2
    (per-army Discrete over a local obs), mirroring watch.py.
    """
    from gymnasium.spaces import Discrete
    from Env.sb3_wrapper import (
        _flat_action_mask, _build_local_obs, N_ACTIONS_PER_ARMY,
    )

    # Tier 2: decide one army at a time from its local observation
    if isinstance(model.action_space, Discrete):
        active   = env._active_army_slots(faction)
        base_obs = env._get_obs(faction)
        legal    = env.legal_actions(faction)
        action   = np.zeros(MAX_ARMIES, dtype=np.int64)
        for slot in range(len(active)):
            obs  = _build_local_obs(env, faction, base_obs, slot)
            mask = np.zeros(N_ACTIONS_PER_ARMY, dtype=bool)
            if slot in legal:
                for d in legal[slot]:
                    mask[d] = True
            else:
                mask[0] = True
            a, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            action[slot] = int(a)
        return action

    # Tier 1: one joint action over the full faction observation
    obs  = env._get_obs(faction)
    mask = _flat_action_mask(env, faction)
    action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
    return np.asarray(action, dtype=np.int64)


def human_action(env: PeninsularWarEnv, faction: int) -> np.ndarray:
    """For each army of the given faction, ask the player to choose a direction."""
    from Env.config import MAX_DEGREE
    legal        = env.legal_actions(faction)
    active_slots = env._active_army_slots(faction)
    action       = np.zeros(MAX_ARMIES, dtype=np.int64)
    name         = 'France' if faction == FRANCE else 'Allies'
    divider      = '─' * 50

    print(f"\n  ── Your armies ({name}) {divider[:max(0, 48 - len(name))]}")
    for slot, dirs in legal.items():
        node_idx = active_slots[slot]
        nid      = env.node_ids[node_idx]
        if faction == FRANCE:
            troops = int(env.france_infantry[node_idx] + env.france_cavalry[node_idx])
            arty   = int(env.france_artillery[node_idx])
            cmd    = env.france_commander[node_idx] or 'UNKNOWN'
        else:
            troops = int(env.allies_infantry[node_idx] + env.allies_cavalry[node_idx])
            arty   = int(env.allies_artillery[node_idx])
            cmd    = env.allies_commander[node_idx] or 'UNKNOWN'

        size_label = f"{troops:>8,} men" + (f"  {arty} guns" if arty else "")
        print(f"\n  Army at {nid}  ({size_label})  cmd={cmd}")
        for d in dirs:
            if d == 0:
                opt = 'Stay'
            elif d <= MAX_DEGREE:
                nbr = env.neighbour_list[nid][d - 1]
                opt = f'Move all  →  {nbr}'
            else:
                nbr = env.neighbour_list[nid][d - MAX_DEGREE - 1]
                opt = f'Split 25% →  {nbr}  (keep 75% here)'
            print(f"    [{d}]  {opt}")

        while True:
            try:
                raw = input(f"  Direction for {nid} (or 'i' for map info): ").strip()
                if raw.lower() == 'i':
                    env.render()
                    continue
                choice = int(raw)
                if choice in dirs:
                    action[slot] = choice
                    break
                print(f"  Invalid — choose from {dirs}")
            except (ValueError, KeyboardInterrupt):
                print("\n  (enter a listed number, or Ctrl-C twice to quit)")

    return action


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Play the Peninsular War against a trained model.")
    ap.add_argument("--play-faction", choices=["france", "allies"], default="france",
                    help="Which faction YOU control (the other is played by the model).")
    ap.add_argument("--tier", type=int, choices=[1, 2], default=2,
                    help="Which trained tier the opponent model comes from (default 2).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed for a reproducible starting position.")
    ap.add_argument("--stochastic", action="store_true",
                    help="Opponent samples its actions instead of taking the argmax.")
    args = ap.parse_args()

    human_faction = FRANCE if args.play_faction == "france" else ALLIES
    model_faction = ALLIES if human_faction == FRANCE else FRANCE
    model_key     = "allies" if model_faction == ALLIES else "france"

    model_path = ROOT / TIER_MODELS[args.tier][model_key]
    if not model_path.exists():
        sys.exit(f"  [play] Tier {args.tier} {model_key} model not found: {model_path}")

    print("Loading opponent model (this can take a moment) ...")
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(model_path), device="cpu")
    deterministic = not args.stochastic

    env = PeninsularWarEnv(render_mode='human')
    obs, info = env.reset(seed=args.seed)

    human_name = 'France' if human_faction == FRANCE else 'Allies'
    model_name = 'Allies' if model_faction == ALLIES else 'France'
    print("═" * 54)
    print("  PENINSULAR WAR  —  Human vs Trained Model")
    print(f"  You play  : {human_name}")
    print(f"  Opponent  : {model_name}  (Tier {args.tier}: {model_path.parent.name})")
    print(f"  Opponent policy: {'stochastic' if args.stochastic else 'deterministic'}")
    print("  One turn ≈ one week.  Max turns: 313 (~6 years)")
    print("═" * 54)
    env.render()

    renderer = MapRenderer(env)

    while True:
        # Build each faction's action: human for the chosen side, model for the other.
        if human_faction == FRANCE:
            f_action = human_action(env, FRANCE)
            a_action = model_action(model, env, ALLIES, deterministic)
        else:
            a_action = human_action(env, ALLIES)
            f_action = model_action(model, env, FRANCE, deterministic)

        actions = {'france': f_action, 'allies': a_action}
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Print battle report
        if info.get('battles'):
            print("\n  ⚔  Battles resolved this turn:")
            for b in info['battles']:
                f_before = b['france_inf_before'] + b['france_cav_before']
                a_before = b['allies_inf_before']  + b['allies_cav_before']
                print(
                    f"     {b['node']:4s}  {b['winner']:<6} wins  |  "
                    f"FR before: {f_before:>7,}  "
                    f"(−{b['france_men_cas']:>6,} men  −{b['france_arty_cas']:>3} guns)   "
                    f"AL before: {a_before:>7,}  "
                    f"(−{b['allies_men_cas']:>6,} men  −{b['allies_arty_cas']:>3} guns)"
                )

        # Update live map
        renderer.update(info.get('battles', []))

        if terminated or truncated:
            if terminated:
                winner = 'France' if info['allies_troops'] == 0 else 'Allies'
                print(f"\n  ★  GAME OVER — {winner} wins after {info['turn']} turns!")
            else:
                f_n = info['france_nodes']
                a_n = info['allies_nodes']
                lead = 'France' if f_n > a_n else ('Allies' if a_n > f_n else 'Draw')
                print(f"\n  ★  Time limit reached ({info['turn']} turns). "
                      f"Score — FR: {f_n} nodes  AL: {a_n} nodes  → {lead} leads.")
            break


if __name__ == '__main__':
    main()
