import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from sb3_contrib import MaskablePPO
from gymnasium.spaces import Discrete
from Env import PeninsularWarEnv
from Env.pygame_renderer import PygameRenderer
from Env.config import FRANCE, ALLIES, MAX_ARMIES
from Env.sb3_wrapper import _flat_action_mask, _build_local_obs, N_ACTIONS_PER_ARMY


DEFAULT_FRANCE = "models/Tier-2/20260606-213718-t2-phase1-france/final.zip"
DEFAULT_ALLIES = "models/Tier-2/20260606-213718-t2-phase2-allies/final.zip"


def model_action(model, env, faction, deterministic=True):
    """
    Query a MaskablePPO model for one faction's full MAX_ARMIES action array.
    """
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

    obs = env._get_obs(faction)
    mask = _flat_action_mask(env, faction)
    action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
    return np.asarray(action, dtype=np.int64)


def print_battles(info):
    if not info.get('battles'):
        return
    print("\n  ⚔  Battles resolved this turn:")
    for b in info['battles']:
        f_before = b['france_inf_before'] + b['france_cav_before']
        a_before = b['allies_inf_before'] + b['allies_cav_before']
        print(
            f"     {b['node']:4s}  {b['winner']:<6} wins  |  "
            f"FR before: {f_before:>7,}  "
            f"(−{b['france_men_cas']:>6,} men  −{b['france_arty_cas']:>3} guns)   "
            f"AL before: {a_before:>7,}  "
            f"(−{b['allies_men_cas']:>6,} men  −{b['allies_arty_cas']:>3} guns)"
        )


def main():
    ap = argparse.ArgumentParser(
        description="Watch the trained agents play, rendered with Pygame.")
    ap.add_argument("--france", default=DEFAULT_FRANCE, help="Path to France model .zip")
    ap.add_argument("--allies", default=DEFAULT_ALLIES, help="Path to Allies model .zip")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="Seconds per turn (playback pace). Default 1.5.")
    ap.add_argument("--seed", type=int, default=None, help="Seed for a reproducible game.")
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample actions instead of taking argmax.")
    ap.add_argument("--reproduce", action="store_true",
                    help="Faithfully replay the simulate_history.py ensemble episode "
                         "with this --seed: forces sampled (stochastic) actions and "
                         "seeds random/numpy/torch/env exactly as the ensemble does.")
    ap.add_argument("--record", default=None, metavar="PATH",
                    help="Save the run to an .mp4 (e.g. --record game.mp4).")
    ap.add_argument("--snapshot", default=None, metavar="PATH",
                    help="Save a PNG of the turn-0 (reset) map and exit, without "
                         "playing (e.g. --snapshot turn0.png).")
    ap.add_argument("--smooth-video", action="store_true",
                    help="With --record, capture every frame (smooth, larger file).")
    args = ap.parse_args()

    france_path = (ROOT / args.france) if not Path(args.france).is_absolute() else Path(args.france)
    allies_path = (ROOT / args.allies) if not Path(args.allies).is_absolute() else Path(args.allies)
    for label, p in (("France", france_path), ("Allies", allies_path)):
        if not p.exists():
            sys.exit(f"  [watch_pygame] {label} model not found: {p}")

    # --reproduce: match simulate_history.py exactly (sampled policy + full seeding)
    deterministic = not (args.stochastic or args.reproduce)
    if args.reproduce and args.seed is None:
        sys.exit("  [watch_pygame] --reproduce needs a --seed (the ensemble episode's seed).")

    print("Loading models (this can take a moment for large checkpoints) ...")
    # Under --reproduce, load on CPU exactly like simulate_history.py
    load_kwargs = {"device": "cpu", "custom_objects": {"n_steps": 1}} if args.reproduce else {}
    france_model = MaskablePPO.load(str(france_path), **load_kwargs)
    allies_model = MaskablePPO.load(str(allies_path), **load_kwargs)

    env = PeninsularWarEnv(render_mode='human')

    if args.reproduce:
        # Same RNG setup and order as simulate_history.run_episode, so a given
        # --seed reproduces that exact ensemble trajectory.
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32 - 1))
        try:
            import torch
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass

    _, info = env.reset(seed=args.seed)

    print("═" * 54)
    print("  PENINSULAR WAR  —  Trained agents, self-play  (Pygame)")
    print(f"  France : {france_path.parent.name}")
    print(f"  Allies : {allies_path.parent.name}")
    print(f"  Policy : {'deterministic' if deterministic else 'stochastic'}"
          f"  |  {args.delay}s / turn")
    print("═" * 54)

    record_path = None
    if args.record:
        record_path = (ROOT / args.record) if not Path(args.record).is_absolute() else Path(args.record)

    renderer = PygameRenderer(
        env, delay=args.delay,
        record=str(record_path) if record_path else None,
        record_every_frame=args.smooth_video,
    )

    # --snapshot: just save the turn-0 (reset) map and exit, no playthrough.
    if args.snapshot:
        snap = Path(args.snapshot)
        snap = snap if snap.is_absolute() else (ROOT / snap)
        renderer.snapshot(snap)
        renderer.close()
        return

    # Save the turn-0 (reset) state as a PNG and make it the first video frame (for report purposes)
    if record_path:
        turn0_png = record_path.with_name(record_path.stem + '_turn0.png')
        renderer.snapshot(turn0_png, add_to_video=True)

    try:
        while not renderer.closed:
            f_action = model_action(france_model, env, FRANCE, deterministic)
            a_action = model_action(allies_model, env, ALLIES, deterministic)

            _, _, terminated, truncated, info = env.step(
                {'france': f_action, 'allies': a_action})

            print_battles(info)
            if not renderer.update(info.get('battles', [])):
                break   # Window closed

            if terminated or truncated:
                if terminated:
                    winner = 'France' if info['allies_troops'] == 0 else 'Allies'
                    print(f"\n  ★  GAME OVER — {winner} wins after {info['turn']} turns!")
                else:
                    f_n, a_n = info['france_nodes'], info['allies_nodes']
                    lead = 'France' if f_n > a_n else ('Allies' if a_n > f_n else 'Draw')
                    print(f"\n  ★  Time limit reached ({info['turn']} turns). "
                          f"Score — FR: {f_n} nodes  AL: {a_n} nodes  → {lead} leads.")
                break
    except KeyboardInterrupt:
        print("\n  (stopped)")
    finally:
        if record_path and not renderer.closed:
            # Flush the video file before the post-game wait
            pass

    if not renderer.closed:
        print("\n  Close the map window to exit.")
        renderer.keep_open()
    renderer.close()


if __name__ == '__main__':
    main()
