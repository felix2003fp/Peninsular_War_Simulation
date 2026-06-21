from __future__ import annotations
from typing import Any, List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .peninsular_war_env import PeninsularWarEnv
from .config import FRANCE, ALLIES, MAX_ARMIES, MAX_DEGREE

# Per-army action cardinality: 0 = stay, 1..MAX_DEGREE = move-all,
# MAX_DEGREE+1..2*MAX_DEGREE = split-25%
N_ACTIONS_PER_ARMY = 2 * MAX_DEGREE + 1   # 17


# Size of one raw per-node block in the full faction observation (Tier 1)
_NODE_BLOCK = 43


# Size of one raw per-node block in the partial observation (Tier 2)

# The 18 "intrinsic" per-node feature offsets we keep (drop [12] (visible), and
# drop the [19-42] neighbour-adjacency embedding — that is re-expressed as
# explicit per-neighbour edge features below)
_INTRINSIC_IDX = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18],
    dtype=np.intp,
)
_INTRINSIC_DIM = int(_INTRINSIC_IDX.size)   # 18

# Per-neighbour: edge (dist, road) + that neighbour's intrinsic features
_EDGE_DIM = 2
_PER_NBR_DIM = _EDGE_DIM + _INTRINSIC_DIM    # 20

# Global faction features appended to every local obs
_GLOBAL_OBS_DIM = 11

# Full local observation: self + 8 neighbours + globals + slot indicator
LOCAL_OBS_DIM = (
    _INTRINSIC_DIM                       # self node
    + MAX_DEGREE * _PER_NBR_DIM          # 8 neighbours × (dist, road, intrinsic)
    + _GLOBAL_OBS_DIM                    # global faction features
    + 1                                  # normalised slot indicator
)                                        # = 18 + 160 + 11 + 1 = 190


# Opponent policies

class RandomMaskedOpponent:
    """Picks a uniformly random LEGAL direction for every army."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def act(self, env: PeninsularWarEnv, faction: int) -> np.ndarray:
        legal = env.legal_actions(faction)
        action = np.zeros(MAX_ARMIES, dtype=np.int64)
        for slot, dirs in legal.items():
            action[slot] = self.rng.choice(dirs)
        return action


class FrozenPolicyOpponent:
    """
    Wraps a Tier-1 MaskablePPO model (MultiDiscrete action space).
    The frozen side observes the env from ITS OWN perspective and acts
    deterministically.
    """

    def __init__(self, model: Any):
        self.model = model

    def act(self, env: PeninsularWarEnv, faction: int) -> np.ndarray:
        obs = env._get_obs(faction)
        mask = _flat_action_mask(env, faction)
        action, _ = self.model.predict(
            obs, action_masks=mask, deterministic=True
        )
        return np.asarray(action, dtype=np.int64)


class FrozenSequentialOpponent:
    """
    Wraps a Tier-2 (LocalObsArmyEnv) MaskablePPO model — Discrete(17) action.

    Because the model decides one army at a time from a local observation, we
    call it once per active army slot and assemble the full MAX_ARMIES action
    array the game engine expects.
    """

    def __init__(self, model: Any):
        self.model = model

    def act(self, env: PeninsularWarEnv, faction: int) -> np.ndarray:
        active = env._active_army_slots(faction)
        n_slots = len(active)
        base_obs = env._get_obs(faction)      # Full faction obs (9901,)
        action = np.zeros(MAX_ARMIES, dtype=np.int64)

        for slot in range(n_slots):
            obs = _build_local_obs(env, faction, base_obs, slot)

            # Legal-action mask for this slot only
            legal = env.legal_actions(faction)
            mask = np.zeros(N_ACTIONS_PER_ARMY, dtype=bool)
            if slot in legal:
                for d in legal[slot]:
                    mask[d] = True
            else:
                mask[0] = True   # only "stay"

            a, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
            action[slot] = int(a)

        return action


# Tier-1: flat action-mask helper

def _flat_action_mask(env: PeninsularWarEnv, faction: int) -> np.ndarray:
    """
    Flat boolean mask of length MAX_ARMIES * N_ACTIONS_PER_ARMY (= 10*17 = 170).
    Layout: mask[slot*17 + d] = True if action d is legal for army slot.

    Inactive slots get action 0 ("stay") marked legal so MaskablePPO always
    has at least one legal choice per dimension.
    """
    legal = env.legal_actions(faction)
    mask = np.zeros(MAX_ARMIES * N_ACTIONS_PER_ARMY, dtype=bool)
    for slot in range(MAX_ARMIES):
        if slot in legal:
            for d in legal[slot]:
                mask[slot * N_ACTIONS_PER_ARMY + d] = True
        else:
            mask[slot * N_ACTIONS_PER_ARMY + 0] = True   # Only "stay"
    return mask


# Tier-2: local obs builder

def _intrinsic(base_obs: np.ndarray, node_idx: int) -> np.ndarray:
    """
    The 18 intrinsic features of one node, sliced from the full faction obs.
    """
    block = base_obs[node_idx * _NODE_BLOCK : (node_idx + 1) * _NODE_BLOCK]
    return block[_INTRINSIC_IDX]


def _build_local_obs(
    env: PeninsularWarEnv,
    faction:  int,
    base_obs: np.ndarray,
    slot: int,
) -> np.ndarray:
    """
    Build the LOCAL observation (LOCAL_OBS_DIM = 190) for the army at `slot`.

    Layout:
    [0 – 17] — self node intrinsic features
    [18 – 177] — 8 neighbour slots, each:
                    dist (1) + road (1) + neighbour intrinsic (18)
                    ordered by move-direction d (1..8); a missing
                    edge contributes all zeros for that slot.
    [178 – 188] — global faction features
    [189] — current slot normalised to [0, 1]
    """
    N = env.N
    active = env._active_army_slots(faction)
    slot_norm = np.array([slot / max(MAX_ARMIES - 1, 1)], dtype=np.float32)
    global_feats = base_obs[N * _NODE_BLOCK : N * _NODE_BLOCK + _GLOBAL_OBS_DIM]

    # No army at this slot -> zero self + neighbours, but real globals and slot id.
    if slot >= len(active):
        body = np.zeros(_INTRINSIC_DIM + MAX_DEGREE * _PER_NBR_DIM, dtype=np.float32)
        return np.concatenate([body, global_feats, slot_norm]).astype(np.float32)

    node_idx = active[slot]
    node_id = env.node_ids[node_idx]
    cur_block = base_obs[node_idx * _NODE_BLOCK : (node_idx + 1) * _NODE_BLOCK]

    parts: List[np.ndarray] = [cur_block[_INTRINSIC_IDX]]   # self intrinsic (18)

    nbrs = env.neighbour_list[node_id]
    for d in range(MAX_DEGREE):
        edge_off = 19 + d * 3        # within the 43-block: (idx, dist, road)
        dist = cur_block[edge_off + 1]
        road = cur_block[edge_off + 2]
        nbr_name = nbrs[d] if d < len(nbrs) else None
        if nbr_name is not None:
            nbr_intr = _intrinsic(base_obs, env.node_idx[nbr_name])
        else:
            nbr_intr = np.zeros(_INTRINSIC_DIM, dtype=np.float32)
        parts.append(np.array([dist, road], dtype=np.float32))
        parts.append(nbr_intr)

    parts.append(global_feats)
    parts.append(slot_norm)
    return np.concatenate(parts).astype(np.float32)



# Tier-1: Single-faction wrapper

class SingleFactionEnv(gym.Env):
    """
    Tier-1 wrapper: exposes ONE faction of PeninsularWarEnv to a single-agent
    SB3 learner.  All MAX_ARMIES army actions are decided in a single step.

    Reuses the process defined in peninsular_war_env.py.

    Parameters
    training_faction : 'france' or 'allies'
    opponent: object with .act(env, faction) -> np.ndarray
    render_mode: passed through to the underlying env
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        training_faction: str = "france",
        opponent: Optional[Any] = None,
        render_mode: Optional[str] = None,
    ):
        assert training_faction in ("france", "allies")
        self.training_faction = training_faction
        self.opponent_faction = "allies" if training_faction == "france" else "france"
        self._train_idx = FRANCE if training_faction == "france" else ALLIES
        self._opp_idx = ALLIES if training_faction == "france" else FRANCE

        self.env = PeninsularWarEnv(render_mode=render_mode, verbose=False)
        self.opponent = opponent or RandomMaskedOpponent()
        self.render_mode = render_mode

        self.observation_space = self.env.observation_space   # Box(9944,)
        self.action_space = self.env.action_space        # MultiDiscrete([17]*10)


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs[self.training_faction], info

    def step(self, action):
        opp_action = self.opponent.act(self.env, self._opp_idx)
        actions = {
            self.training_faction: np.asarray(action, dtype=np.int64),
            self.opponent_faction: np.asarray(opp_action, dtype=np.int64),
        }
        obs, rewards, terminated, truncated, info = self.env.step(actions) # Follows the process of oeninsular_war_env.py
        return (
            obs[self.training_faction],
            float(rewards[self.training_faction]),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        return self.env.render()

    # For MaskablePPO

    def action_masks(self) -> np.ndarray:
        """
        Flat 170-element bool mask (10 armies × 17 actions).
        Called by sb3-contrib's ActionMasker before each prediction.
        """
        return _flat_action_mask(self.env, self._train_idx)

    def set_opponent(self, opponent: Any) -> None:
        """
        Hot-swap the frozen opponent (used between training phases).
        """
        self.opponent = opponent


# Tier-2: Local-obs sequential wrapper

class LocalObsArmyEnv(gym.Env):
    """
    Tier-2 wrapper: one SB3 step = one army's decision, from a LOCAL view.

    The training faction's armies submit their actions one at a time, each from
    an observation centred on that army's node (see _build_local_obs).  Only
    after the last active army has decided does the game turn resolve (both
    factions' actions applied, battles fought, reward computed).

    Observation space
    Box(LOCAL_OBS_DIM,) = Box(190,).  See _build_local_obs for the layout.

    Action space
    Discrete(17) — masked to legal moves for the CURRENT army only.

    Reward
    0.0 for all intermediate sub-steps within a turn.
    Full game-turn reward on the last sub-step (when the game actually fires).

    Parameters
    training_faction: 'france' or 'allies'
    opponent: object with .act(env, faction) -> np.ndarray
    render_mode: passed through to the underlying env
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        training_faction: str = "france",
        opponent: Optional[Any] = None,
        render_mode: Optional[str] = None,
    ):
        assert training_faction in ("france", "allies")
        self.training_faction = training_faction
        self.opponent_faction = "allies" if training_faction == "france" else "france"
        self._train_idx = FRANCE if training_faction == "france" else ALLIES
        self._opp_idx = ALLIES if training_faction == "france" else FRANCE

        self.env = PeninsularWarEnv(render_mode=render_mode, verbose=False)
        self.opponent = opponent or RandomMaskedOpponent()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape = (LOCAL_OBS_DIM,),   # 190
            dtype = np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS_PER_ARMY)   # 17

        # Sub-step state (reset at episode start and after each game turn)
        _base_dim = self.env.observation_space.shape[0]   # 9901
        self._base_obs: np.ndarray = np.zeros(_base_dim, dtype=np.float32)
        self._buffered_actions: np.ndarray = np.zeros(MAX_ARMIES, dtype=np.int64)
        self._active_slots: List[int] = []
        self._current_slot: int = 0



    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._base_obs = obs[self.training_faction].copy()
        self._active_slots = list(self.env._active_army_slots(self._train_idx))
        self._current_slot = 0
        self._buffered_actions = np.zeros(MAX_ARMIES, dtype=np.int64)
        return self._make_obs(), info

    def step(self, action):
        n_active = len(self._active_slots)

        # Buffer this army's choice (if there are any active armies)
        if n_active > 0 and self._current_slot < n_active:
            self._buffered_actions[self._current_slot] = int(action)
            self._current_slot += 1

        # If more armies still need to decide, return an intermediate step
        if self._current_slot < n_active:
            return self._make_obs(), 0.0, False, False, {}

        # All armies decided (or no armies) — fire the game turn
        opp_action = self.opponent.act(self.env, self._opp_idx)
        game_actions = {
            self.training_faction: self._buffered_actions.copy(),
            self.opponent_faction: np.asarray(opp_action, dtype=np.int64),
        }
        obs, rewards, terminated, truncated, info = self.env.step(game_actions)

        # Reset sub-step state for the next turn
        self._base_obs = obs[self.training_faction].copy()
        self._active_slots = list(self.env._active_army_slots(self._train_idx))
        self._current_slot = 0
        self._buffered_actions = np.zeros(MAX_ARMIES, dtype=np.int64)

        reward = float(rewards[self.training_faction])
        return self._make_obs(), reward, bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render()

    # For MaskablePPO

    def action_masks(self) -> np.ndarray:
        """
        17-element bool mask for the CURRENT army slot only.
        Called by sb3-contrib's ActionMasker before each prediction.
        """
        legal = self.env.legal_actions(self._train_idx)
        slot = self._current_slot
        mask = np.zeros(N_ACTIONS_PER_ARMY, dtype=bool)
        if slot in legal:
            for d in legal[slot]:
                mask[d] = True
        else:
            mask[0] = True   # "stay" only — inactive / past-last-slot fallback
        return mask

    def set_opponent(self, opponent: Any) -> None:
        self.opponent = opponent


    def _make_obs(self) -> np.ndarray:
        """
        Local obs for the current army (self + neighbours + globals + slot).
        """
        return _build_local_obs(
            env = self.env,
            faction = self._train_idx,
            base_obs = self._base_obs,
            slot = self._current_slot,
        )
