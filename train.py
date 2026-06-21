from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Ensure TFG/ is on sys.path so `from Env import ...` works regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from Env.sb3_wrapper import (
    SingleFactionEnv,
    LocalObsArmyEnv,
    RandomMaskedOpponent,
    FrozenPolicyOpponent,
    FrozenSequentialOpponent,
)


# W&B callback


class WandbMetricsCallback(BaseCallback):
    """
    Sends all of SB3's training scalars (loss, entropy, KL, explained variance,
    ep_rew_mean, ep_len_mean, fps …) to W&B, and handles periodic checkpoints.

    We wrap the logger's dump() method directly. dump() is called once per PPO 
    update with every scalar populated; we snapshot the logger's name_to_value 
    buffer at that exact moment (before SB3 clears it) and forward it to wandb.log(). 
    This can't be bypassed by output-format quirks.

    Logged keys (the same set SB3 would send to TensorBoard):
        train/policy_gradient_loss, train/value_loss, train/entropy_loss,
        train/loss, train/approx_kl, train/clip_fraction,
        train/explained_variance, rollout/ep_rew_mean, rollout/ep_len_mean,
        time/fps  (+ a few timestep counters)
    All bound to train/global_step as their x-axis.
    """

    def __init__(self, model_save_path: str, model_save_freq: int = 50_000,
                 verbose: int = 0):
        super().__init__(verbose)
        self._save_path = Path(model_save_path)
        self._save_freq = model_save_freq
        self._orig_dump = None

    def _on_training_start(self) -> None:
        import wandb
        # Bind every training scalar to total_timesteps as its x-axis, so these
        # curves are unaffected by the global step advanced by the per-step
        # episode reward / endep logging.
        wandb.define_metric("train/global_step")
        for prefix in ("train/*", "rollout/*", "time/*"):
            wandb.define_metric(prefix, step_metric="train/global_step")

        logger = self.model.logger
        if getattr(logger, "_wandb_patched", False):
            return

        orig_dump = logger.dump

        def patched_dump(step: int = 0, _orig=orig_dump, _logger=logger):
            import wandb as _wandb
            # Snapshot BEFORE the original dump clears the buffers.
            name_to_value = dict(_logger.name_to_value)
            name_to_excluded = dict(_logger.name_to_excluded)
            _orig(step)   # let SB3 do its normal stdout flush + clear

            payload = {}
            for k, v in name_to_value.items():
                excl = name_to_excluded.get(k)
                # Mimic the TensorBoard writer: skip keys excluded from it
                # (e.g. time/total_timesteps), keep everything else numeric.
                if excl is not None and "tensorboard" in excl:
                    continue
                # Accept any numeric scalar, incl. NumPy types (np.float32 is
                # NOT a subclass of float, so isinstance(...) silently dropped
                # approx_kl etc.). bool is excluded; only real numbers pass.
                if isinstance(v, bool):
                    continue
                try:
                    payload[k] = float(v)
                except (TypeError, ValueError):
                    continue
            if payload:
                payload["train/global_step"] = float(step)
                _wandb.log(payload)

        logger.dump = patched_dump
        logger._wandb_patched = True
        self._orig_dump = orig_dump

    def _on_step(self) -> bool:
        import wandb
        # Periodic model snapshot
        if self._save_freq > 0 and self.num_timesteps % self._save_freq == 0:
            self._save_path.mkdir(parents=True, exist_ok=True)
            ckpt = str(self._save_path / f"step_{self.num_timesteps}.zip")
            self.model.save(ckpt)
            wandb.save(ckpt)
        return True

    def _on_training_end(self) -> None:
        import wandb
        logger = self.model.logger
        # SB3 calls dump() BEFORE train(), so the final rollout's train/* scalars
        # are recorded after the last dump and never flushed (lost in SB3's own
        # TensorBoard output too).
        payload = {}
        for k, v in dict(logger.name_to_value).items():
            excl = logger.name_to_excluded.get(k)
            if excl is not None and "tensorboard" in excl:
                continue
            if isinstance(v, bool):
                continue
            try:
                payload[k] = float(v)
            except (TypeError, ValueError):
                continue
        if payload:
            payload["train/global_step"] = float(self.num_timesteps)
            wandb.log(payload)

        # Restore the original dump so the patched closure isn't reused.
        if self._orig_dump is not None:
            logger.dump = self._orig_dump
            logger._wandb_patched = False
            self._orig_dump = None



class EpisodeEndStatsCallback(BaseCallback):
    """
    Logs end-of-episode game state for env 0 to W&B, aggregated PER ROLLOUT.

    Within a rollout, every env-0 episode that ends has its terminal stats
    collected. At the end of the rollout we log the MEAN of each metric across
    those episodes as a single point, keyed to the rollout number, e.g.
    endep/france_troops, endep/allies_troops, ...

    One value per rollout means each metric is one clean series over rollouts:
    render it as a BAR chart and each rollout is a single bar. This is the right
    shape because the episodes inside a rollout are independent samples of the
    SAME policy — averaging them is meaningful, connecting them with a line is
    not.

    For BOTH factions, at each env-0 episode end it collects (then averages):
      - total troops (infantry + cavalry + artillery) -> france_troops / allies_troops
      - number of active armies -> france_armies / allies_armies
      - nodes controlled, split by node type -> {faction}_nodes_{type}
        (capital, regional_capital, major_city, city, town, intersection)
      - episode length in turns -> turn

    Values come from the terminal-step info dict produced by
    PeninsularWarEnv._get_info().
    """

    _NODE_TYPES = ('capital', 'regional_capital', 'major_city',
                   'city', 'town', 'intersection')

    def __init__(self, csv_path: str | None = None, use_wandb: bool = True,
                 verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.use_wandb = use_wandb
        self._rollout_idx = 0
        self._game_turns = 0          # cumulative RESOLVED game turns (all envs)
        self._acc: dict[str, list[float]] = defaultdict(list)  # field -> values this rollout
        self._fh = None
        self._writer = None

        # Per-rollout game-state fields (keys map to PeninsularWarEnv._get_info()).
        self._ENDEP_FIELDS = (
            ["turn", "france_troops", "allies_troops",
             "france_armies", "allies_armies"]
            + [f"france_nodes_{t}" for t in self._NODE_TYPES]
            + [f"allies_nodes_{t}" for t in self._NODE_TYPES]
        )
        self._CSV_FIELDS = (
            ["rollout", "timesteps", "game_turns", "episodes",
             "ep_rew_mean", "ep_len_mean"]
            + self._ENDEP_FIELDS
        )

    def _ep_info_means(self):
        """
        Mean episode reward / length over SB3's buffer (all envs).
        """
        buf = getattr(self.model, "ep_info_buffer", None)
        if buf and len(buf) > 0:
            return (float(np.mean([e["r"] for e in buf])),
                    float(np.mean([e["l"] for e in buf])))
        return float("nan"), float("nan")

    def _on_training_start(self) -> None:
        self._rollout_idx = 0
        self._game_turns = 0
        self._acc = defaultdict(list)

        if self.csv_path:
            p = Path(self.csv_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(p, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._CSV_FIELDS)
            self._writer.writeheader()
            self._fh.flush()

        if self.use_wandb:
            import wandb
            # Cumulative game turns is the cross-tier x-axis (Tier 1: 1 turn/step;
            # Tier 2: n_armies steps/turn). Bind per-rollout means to it so the
            # W&B panels can be plotted directly against game turns.
            wandb.define_metric("endep/game_turns")
            wandb.define_metric("endep/rollout")
            wandb.define_metric("endep/mean/*", step_metric="endep/game_turns")
            wandb.define_metric("endep/ep_rew_mean", step_metric="endep/game_turns")
            wandb.define_metric("endep/ep_len_mean", step_metric="endep/game_turns")

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        # Count every RESOLVED game turn across all envs. A resolved turn carries
        # the env's real info dict (always has "turn"); Tier-2 intermediate
        # sub-steps return an empty info and are skipped — so this counts game
        # WEEKS, the cross-tier experience axis, not SB3 sub-steps.
        for inf in infos:
            if "turn" in inf:
                self._game_turns += 1

        # Collect end-of-episode game state from every env that just finished.
        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i]
            if "france_troops_total" not in info:   # only enriched info
                continue
            self._acc["turn"].append(info.get("turn", 0))
            self._acc["france_troops"].append(info["france_troops_total"])
            self._acc["allies_troops"].append(info["allies_troops_total"])
            self._acc["france_armies"].append(info["france_armies"])
            self._acc["allies_armies"].append(info["allies_armies"])
            for t in self._NODE_TYPES:
                self._acc[f"france_nodes_{t}"].append(info[f"france_nodes_{t}"])
                self._acc[f"allies_nodes_{t}"].append(info[f"allies_nodes_{t}"])
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_idx += 1
        ep_rew, ep_len = self._ep_info_means()
        n_eps = len(self._acc["turn"]) if "turn" in self._acc else 0

        row = {
            "rollout": self._rollout_idx,
            "timesteps": int(self.num_timesteps),
            "game_turns": int(self._game_turns),
            "episodes": n_eps,
            "ep_rew_mean": ep_rew,
            "ep_len_mean": ep_len,
        }
        for f in self._ENDEP_FIELDS:
            vals = self._acc.get(f)
            row[f] = (sum(vals) / len(vals)) if vals else ""

        if self._writer is not None:
            self._writer.writerow(row)
            self._fh.flush()

        if self.use_wandb:
            import wandb
            payload = {
                "endep/rollout": self._rollout_idx,
                "endep/game_turns": self._game_turns,
                "endep/ep_rew_mean": ep_rew,
                "endep/ep_len_mean": ep_len,
                "endep/mean/episodes": n_eps,
            }
            for f in self._ENDEP_FIELDS:
                vals = self._acc.get(f)
                if vals:
                    payload[f"endep/mean/{f}"] = sum(vals) / len(vals)
            wandb.log(payload)

        self._acc = defaultdict(list)

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None


class BestRewardCallback(BaseCallback):
    """
    Keeps the checkpoint with the highest smoothed episode reward, so a phase's
    output is its best-performing model rather than whatever the last gradient
    step produced. Near-zero overhead: it reuses the episode-reward buffer SB3
    already maintains (the same ep_rew_mean logged to W&B), so no extra games are
    played.

    At each rollout end it reads the mean reward over SB3's ep_info_buffer (last
    100 episodes across all envs, under the current — still exploring — policy
    against the fixed frozen opponent). When that mean improves, the model is
    saved to best.zip.  
    """

    def __init__(self, save_dir: str, use_wandb: bool, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.use_wandb = use_wandb
        self.best_reward = -float("inf")
        self.best_path: str | None = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = getattr(self.model, "ep_info_buffer", None)
        if not buf or len(buf) == 0:
            return
        mean_r = float(np.mean([e["r"] for e in buf]))
        if mean_r > self.best_reward:
            self.best_reward = mean_r
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.best_path = str(self.save_dir / "best.zip")
            self.model.save(self.best_path)
            if self.verbose:
                print(f"  [best] ep_rew_mean={mean_r:8.3f} @ "
                      f"{self.num_timesteps:>10,} steps — saved best.zip")
            if self.use_wandb:
                import wandb
                wandb.log({"best/ep_rew_mean": self.best_reward,
                           "train/global_step": self.num_timesteps})


# Network architecture defaults (per tier)

_DEFAULT_ARCH: dict[int, list[int]] = {
    1: [2048, 1024, 512],   # Tier 1: gradual funnel from 9901 input dims
    2: [512, 256],          # Tier 2: local obs is only 190 dims -> smaller net
}


# Env factory

def _mask_fn(env):
    """
    ActionMasker calls this each step to get the current legal-action mask.
    """
    return env.action_masks()


def make_env(training_faction: str, opponent, seed: int, tier: int):
    """
    Returns a thunk (zero-arg callable) that builds ONE wrapped env instance.
    DummyVecEnv takes a list of these thunks and builds the parallel batch.

    tier 1 -> SingleFactionEnv (MultiDiscrete([17]*10), full obs 9901 dims)
    tier 2 -> LocalObsArmyEnv (Discrete(17), local obs 190 dims)
    """
    def _init():
        if tier == 1:
            env = SingleFactionEnv(training_faction=training_faction, opponent=opponent)
        else:
            env = LocalObsArmyEnv(training_faction=training_faction, opponent=opponent)

        env = ActionMasker(env, _mask_fn)
        env = Monitor(env)          # records episode reward / length
        env.reset(seed=seed)
        return env
    return _init


# A single training phase

def train_phase(
    *,
    phase_idx:       int,
    training_faction: str,
    opponent,
    total_timesteps: int,
    n_envs:          int,
    tier:            int,
    arch:            list[int],
    run_name:        str,
    use_wandb:       bool,
    init_from:       str | None = None,
    select_best:     bool = True,
) -> str:
    """
    Runs one phase of iterated best-response and returns the saved model path.

    Parameters
    init_from : path to a previous model checkpoint for warm-starting.
                Only meaningful when the tier (and therefore obs/action shapes)
                matches — which it always does within a single run.
    """
    print(f"\n{'═' * 66}")
    print(f"  PHASE {phase_idx}  —  training {training_faction.upper()}  "
          f"against {type(opponent).__name__}")
    print(f"  Tier {tier}  |  net_arch={arch}")
    print(f"  {total_timesteps:,} timesteps × {n_envs} parallel envs")
    print(f"  run name: {run_name}")
    print(f"{'═' * 66}\n")

    # Vectorised env
    vec_env = DummyVecEnv([
        make_env(training_faction, opponent, seed=phase_idx * 1000 + i, tier=tier)
        for i in range(n_envs)
    ])

    # Model
    policy_kwargs = dict(net_arch=arch)

    if init_from and os.path.exists(init_from):
        print(f"  ↳ warm-starting from {init_from}")
        model = MaskablePPO.load(init_from, env=vec_env, device="cpu")
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            vec_env,
            device          = "cpu",
            policy_kwargs   = policy_kwargs,
            n_steps         = 2048,      # Transitions per env per rollout.
            batch_size      = 256,       # Mini-batch size for gradient update
            n_epochs        = 5,         # PPO update passes per rollout
            learning_rate   = 1e-4,      
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            target_kl       = 0.05,      # Stop a rollout's epochs early once the policy drifts this far
            ent_coef        = 0.01,
            verbose         = 1,
        )

    # Callbacks
    save_dir  = Path("./models") / f"Tier-{tier}" / run_name
    csv_path  = f"./models/Tier-{tier}/{run_name}/metrics.csv"
    callbacks = [EpisodeEndStatsCallback(csv_path=csv_path, use_wandb=use_wandb)]
    if use_wandb:
        callbacks.insert(0, WandbMetricsCallback(
            model_save_path = str(save_dir),
            model_save_freq = 50_000,
            verbose         = 1,
        ))

    # Best-checkpoint selection by smoothed episode reward (the same ep_rew_mean
    # logged to W&B)
    best_cb = None
    if select_best:
        best_cb = BestRewardCallback(save_dir=str(save_dir), use_wandb=use_wandb)
        callbacks.append(best_cb)

    # Train
    model.learn(
        total_timesteps     = total_timesteps,
        callback            = callbacks if callbacks else None,
        reset_num_timesteps = True,
    )

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "final.zip"

    # final.zip = the highest-reward checkpoint seen during the phase (which may
    # be the last one). best.zip is also left in place for reference.
    if best_cb is not None and best_cb.best_path is not None:
        print(f"  ↳ keeping best checkpoint (ep_rew_mean={best_cb.best_reward:.3f})")
        model = MaskablePPO.load(best_cb.best_path, device="cpu")

    model.save(str(save_path))
    print(f"\n  ✔ saved → {save_path}")
    return str(save_path)


# Driver

def main():
    ap = argparse.ArgumentParser(
        description="Train MaskablePPO on the Peninsular War environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--tier", type=int, default=1, choices=[1, 2],
        help=(
            "Training tier:\n"
            "  1 = joint action, full map obs (all armies at once, default)\n"
            "  2 = sequential per-army decisions, local obs (fastest to learn)"
        ),
    )
    ap.add_argument(
        "--arch", type=str, default=None,
        metavar="LAYERS",
        help=(
            "Comma-separated hidden layer sizes, e.g. '1024,512,256'.\n"
            "Defaults: tier 1 → '2048,1024,512';  tier 2 → '512,256'."
        ),
    )
    ap.add_argument(
        "--steps", type=int, default=500_000,
        help="Timesteps per phase (default 500k). Used for every phase unless "
             "--phase-steps is given.",
    )
    ap.add_argument(
        "--phase-steps", default=None, metavar="N1,N2,...",
        help=("Per-phase timesteps as a comma-separated list, one value per "
              "phase in run order, e.g. '300000,200000' for a 2-phase run. "
              "Overrides --steps; if fewer values than --phases are given, the "
              "last value is reused for the remaining phases."),
    )
    ap.add_argument(
        "--n-envs", type=int, default=8,
        help="Parallel envs in DummyVecEnv (default 8).",
    )
    ap.add_argument(
        "--phases", type=int, default=2,
        help="Number of alternating iterated best-response phases (default 2).",
    )
    ap.add_argument(
        "--start-side", choices=["france", "allies"], default="france",
        help="Which faction to train first (default france).",
    )
    ap.add_argument(
        "--france-init", default=None, metavar="PATH",
        help=("Existing FRANCE model to seed the run with: warm-starts France "
              "and/or serves as the frozen opponent when training Allies. Use to "
              "continue from earlier phases instead of starting from scratch."),
    )
    ap.add_argument(
        "--allies-init", default=None, metavar="PATH",
        help="Existing ALLIES model to seed the run with (counterpart to --france-init).",
    )
    ap.add_argument(
        "--start-phase", type=int, default=1,
        help=("Phase number to label the first phase of this run (default 1). "
              "E.g. --start-phase 3 to continue an alternating schedule."),
    )
    ap.add_argument(
        "--no-select-best", action="store_true",
        help=("Disable best-checkpoint selection. By default each phase saves "
              "final.zip as the checkpoint with the highest smoothed episode "
              "reward seen during the phase, not just the last gradient step."),
    )
    ap.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging (no metrics will be recorded).",
    )
    ap.add_argument(
        "--wandb-project", default="peninsular-war-marl",
        help="W&B project name.",
    )
    args = ap.parse_args()

    # Resolve network architecture
    if args.arch:
        arch = [int(x.strip()) for x in args.arch.split(",")]
    else:
        arch = _DEFAULT_ARCH[args.tier]

    print(f"\n  Tier {args.tier}  |  net_arch={arch}  |  "
          f"{args.n_envs} envs  |  {args.steps:,} steps/phase\n")

    # Quick sanity check on the env before burning compute
    print("  Running stable_baselines3 env_checker …")
    if args.tier == 1:
        probe = SingleFactionEnv(training_faction=args.start_side)
    else:
        probe = LocalObsArmyEnv(training_faction=args.start_side)
    probe = ActionMasker(probe, _mask_fn)
    check_env(probe, warn=True)
    print("  ✔ env_checker passed\n")
    del probe

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

    # Phase loop (iterated best-response)
    side = args.start_side
    other_side = "allies" if side == "france" else "france"
    # Track each side's most recent model separately so we can:
    #   - use the OTHER side's latest model as the frozen opponent, and
    #   - warm-start THIS side from its OWN latest model (so e.g. phase-3
    #     France continues learning from phase-1 France instead of restarting)
    model_by_side: dict[str, str | None] = {
        "france": args.france_init,
        "allies": args.allies_init,
    }
    for _f, _p in model_by_side.items():
        if _p is not None:
            if not os.path.exists(_p):
                sys.exit(f"  [train] --{_f}-init model not found: {_p}")
            print(f"  Seeding {_f.upper()} from {_p}")
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Per-phase step counts: --phase-steps overrides --steps; if shorter than
    # --phases, the last value repeats for the remaining phases
    if args.phase_steps:
        phase_steps_list = [int(s.strip()) for s in args.phase_steps.split(",")]
    else:
        phase_steps_list = None

    for phase_offset in range(args.phases):
        phase_idx = args.start_phase + phase_offset
        run_name  = f"{timestamp}-t{args.tier}-phase{phase_idx}-{side}"
        steps_this_phase = (
            phase_steps_list[min(phase_offset, len(phase_steps_list) - 1)]
            if phase_steps_list else args.steps
        )

        # Build opponent from the OTHER side's most recent model
        opponent_path = model_by_side[other_side]
        if opponent_path is None:
            opponent = RandomMaskedOpponent()
        else:
            frozen = MaskablePPO.load(opponent_path)
            if args.tier == 1:
                # Tier-1 model outputs MultiDiscrete([17]*10) all at once
                opponent = FrozenPolicyOpponent(frozen)
            else:
                # Tier-2 model outputs Discrete(17) one army at a time from a
                # local obs; FrozenSequentialOpponent calls it per slot to build
                # the full MAX_ARMIES action array the game engine needs.
                opponent = FrozenSequentialOpponent(frozen)

        # Warm-start this side from its own previous model, if any.
        own_init = model_by_side[side]
        if own_init is not None:
            print(f"  ↳ continuing {side.upper()} from {own_init}")

        if use_wandb:
            wandb_run = wandb.init(
                project = args.wandb_project,
                name = run_name,
                config = dict(
                    tier = args.tier,
                    phase = phase_idx,
                    training_faction = side,
                    opponent_type = type(opponent).__name__,
                    total_timesteps = steps_this_phase,
                    n_envs = args.n_envs,
                    net_arch = arch,
                    algo = "MaskablePPO",
                ),
                monitor_gym = False,
                save_code = True,
                reinit = True,
            )

        new_model_path = train_phase(
            phase_idx = phase_idx,
            training_faction = side,
            opponent = opponent,
            total_timesteps = steps_this_phase,
            n_envs = args.n_envs,
            tier = args.tier,
            arch = arch,
            run_name = run_name,
            use_wandb = use_wandb,
            init_from = own_init,
            select_best = not args.no_select_best,
        )
        model_by_side[side] = new_model_path

        if use_wandb:
            wandb_run.finish()

        # Swap sides for next phase
        side, other_side = other_side, side

    print("\n  All phases complete.")
    print(f"  France model: {model_by_side['france']}")
    print(f"  Allies model: {model_by_side['allies']}")


if __name__ == "__main__":
    main()
