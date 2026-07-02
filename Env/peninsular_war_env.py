import heapq
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .config import (
    FRANCE, ALLIES, NEUTRAL,
    SUBFACTION_NONE, SUBFACTION_BRITISH, SUBFACTION_SPANISH, SUBFACTION_PORTUGUESE,
    MAX_TURNS, DOMINANCE_TROOP_FLOOR, FOG_RADIUS_KM, MAX_DEGREE,
    INITIAL_ARMY_SIZE, FRANCE_START_NODES, ALLIES_START_NODES,
    FRANCE_START_SIZES, ALLIES_START_SIZES,
    ALLIES_START_SUBFACTIONS, FRANCE_START_COMMANDERS, ALLIES_START_COMMANDERS,
    NODES_FILE, EDGES_FILE,
    MOMNTA_WINDOW, INITA_WINDOW,
    SUPPLY_DEPOT_TYPES,
    SIEGE_GARRISON_LOSS_RATE, SIEGE_ATTACKER_LOSS_RATE, MAX_GARRISON_OBS,
    COMMANDER_QUALITY, COMMANDER_QUALITY_DEFAULT,
    COMMANDER_SENIORITY, COMMANDER_SENIORITY_DEFAULT,
    CAVALRY_RATIO, ARTY_PER_1000,
    CAVALRY_CASUALTY_RATIO,
    MARCH_SECONDARY_THRESHOLD, MARCH_SECONDARY_RATE,
    MARCH_TERTIARY_THRESHOLD, MARCH_TERTIARY_RATE,
    W_NODE_DELTA, W_CASUALTY, W_BATTLE_OUTCOME,
    W_TERRITORY_CONSOL, W_ARMY_CONSOL, W_ENEMY_DEPOT,
    W_DEPOT_BASE_PENALTY, W_DEPOT_SPIKE,
    NODE_CAPTURE_VALUE,
    COMMANDER_DEATH_PROB,
    FRANCE_COMMANDER_POOL, ALLIES_COMMANDER_POOL,
    HISTORICAL_COMMANDER_EVENTS,
    FRANCE_REINF_DEPOTS, ALLIES_REINF_DEPOTS, ALLIES_REINF_SUBFACTIONS,
    REINF_SPAWN_THRESHOLD, REINF_POOL_CAP, REINFORCEMENT_SCHEDULE,
    MAX_ARMIES, MIN_ARMY_SIZE,
)
from .battle_model import (
    resolve_battle
)


# Helpers

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _subfaction_norm(sf: int) -> float:
    return {SUBFACTION_NONE: 0.0,
            SUBFACTION_BRITISH: 0.33,
            SUBFACTION_SPANISH: 0.67,
            SUBFACTION_PORTUGUESE: 1.0}.get(sf, 0.0)





class PeninsularWarEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode: Optional[str] = None, verbose: bool = True):
        super().__init__()
        self.render_mode = render_mode
        self.verbose = verbose

        root = Path(__file__).parent.parent
        nodes_df = pd.read_csv(root / NODES_FILE)
        edges_df = pd.read_csv(root / EDGES_FILE)

        # Node index
        self.node_ids: List[str] = list(nodes_df['node_id'])
        self.node_idx: Dict[str, int] = {nid: i for i, nid in enumerate(self.node_ids)}
        self.N = len(self.node_ids)

        self._lat: Dict[str, float] = nodes_df.set_index('node_id')['latitude'].to_dict()
        self._lon: Dict[str, float] = nodes_df.set_index('node_id')['longitude'].to_dict()

        # Static terrain
        if 'terra1' in nodes_df.columns:
            self._terra1: Dict[str, str] = nodes_df.set_index('node_id')['terra1'].to_dict()
            self._terra2: Dict[str, str] = nodes_df.set_index('node_id')['terra2'].to_dict()
        else:
            self._terra1 = {nid: 'R' for nid in self.node_ids}
            self._terra2 = {nid: 'M' for nid in self.node_ids}

        # Garrison capacity per node
        if 'garrison_size' in nodes_df.columns:
            self._garrison_cap = np.array(
                [int(nodes_df.set_index('node_id')['garrison_size'].get(nid, 0))
                 for nid in self.node_ids], dtype=np.int32)
        else:
            self._garrison_cap = np.zeros(self.N, dtype=np.int32)

        # Supply depot set (node indices)
        ntype_col = nodes_df.set_index('node_id')['node_type'] if 'node_type' in nodes_df.columns else {}
        self._supply_depots = frozenset(
            self.node_idx[nid] for nid in self.node_ids
            if ntype_col.get(nid, '') in SUPPLY_DEPOT_TYPES
        )

        self._ntype: Dict[str, str] = (
            nodes_df.set_index('node_id')['node_type'].to_dict()
            if 'node_type' in nodes_df.columns else {nid: 'town' for nid in self.node_ids}
        )

        # Strategic importance per node (static, normalised to [0,1])
        if 'strategic_importance' in nodes_df.columns:
            si_map = nodes_df.set_index('node_id')['strategic_importance']
            self._strategic_importance = np.array(
                [float(si_map.get(nid, 1)) / 6.0 for nid in self.node_ids],
                dtype=np.float32,
            )
        else:
            self._strategic_importance = np.full(self.N, 1.0 / 6.0, dtype=np.float32)

        # Dominance check arrays (for victory condition)
        # Un-normalised strategic importance per node
        if 'strategic_importance' in nodes_df.columns:
            si_raw = nodes_df.set_index('node_id')['strategic_importance']
            self._si_raw = np.array(
                [int(si_raw.get(nid, 1)) for nid in self.node_ids], dtype=np.int32
            )
        else:
            self._si_raw = np.ones(self.N, dtype=np.int32)

        # Node-type weight for terminal scoring
        _TW = {'capital': 4.0, 'regional_capital': 2.0, 'major_city': 1.0,
               'city': 0.2, 'town': 0.05, 'intersection': 0.0}
        self._terminal_weight = np.array(
            [_TW.get(self._ntype.get(nid, 'town'), 0.0) for nid in self.node_ids],
            dtype=np.float32,
        )

        # Per-node capture value for the node-delta reward term
        self._capture_value = np.array(
            [NODE_CAPTURE_VALUE.get(self._ntype.get(nid, 'town'), 0.0)
             for nid in self.node_ids],
            dtype=np.float32,
        )

        # Boolean mask per node-type
        self._NODE_TYPES = ('capital', 'regional_capital', 'major_city',
                             'city', 'town', 'intersection')
        self._type_masks = {
            t: np.array([self._ntype.get(nid, 'town') == t for nid in self.node_ids], dtype=bool)
            for t in self._NODE_TYPES
        }

        # Counts of nodes by SI tier (used in dominance check)
        self._n_si_high = int(np.sum(self._si_raw >= 4))   # SI = 4,5,6
        self._n_si_three = int(np.sum(self._si_raw == 3))   # SI = 3
        self._n_si_two = int(np.sum(self._si_raw == 2))   # SI = 2

        # Reinforcement depot node-index sets (used by observation builder)
        self._france_depot_nodes: frozenset = frozenset(
            self.node_idx[nid] for nid in FRANCE_REINF_DEPOTS if nid in self.node_idx
        )
        self._allies_depot_nodes: frozenset = frozenset(
            self.node_idx[nid] for nid in ALLIES_REINF_DEPOTS if nid in self.node_idx
        )

        # Adjacency
        raw_nbrs: Dict[str, List[str]] = {nid: [] for nid in self.node_ids}
        for _, row in edges_df.iterrows():
            n1, n2 = row['node1'], row['node2']
            if n1 in raw_nbrs and n2 in raw_nbrs:
                raw_nbrs[n1].append(n2)
                raw_nbrs[n2].append(n1)

        raw_dist: Dict[tuple, float] = {}
        for _, row in edges_df.iterrows():
            d = float(row.get('distance_km', 1.0))
            raw_dist[(row['node1'], row['node2'])] = d
            raw_dist[(row['node2'], row['node1'])] = d

        self.neighbour_list: Dict[str, List[Optional[str]]] = {}
        self.neighbour_dist: Dict[str, List[Optional[float]]] = {}
        for nid in self.node_ids:
            nbrs = raw_nbrs[nid][:MAX_DEGREE]
            dists = [raw_dist.get((nid, nbr), 1.0) for nbr in nbrs]
            self.neighbour_list[nid] = nbrs  + [None] * (MAX_DEGREE - len(nbrs))
            self.neighbour_dist[nid] = dists + [None] * (MAX_DEGREE - len(dists))

        self._road_type: Dict[Tuple[str, str], str] = {}
        for _, row in edges_df.iterrows():
            rt = row.get('road_type', 'primary')
            self._road_type[(row['node1'], row['node2'])] = rt
            self._road_type[(row['node2'], row['node1'])] = rt

        # Initializing the observation space
        # Per-node: 19 base values + 8 neighbours × 3 = 43 total
        # Global:   11 values
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.N * 43 + 11,),
            dtype=np.float32,
        )

        # Initializing the Action Space
        # Per army, 17 movements are possible (before masking)

        self.action_space = spaces.MultiDiscrete([2 * MAX_DEGREE + 1] * MAX_ARMIES)

        self.owner = np.full(self.N, NEUTRAL, dtype=np.int8)
        self.france_infantry  = np.zeros(self.N, dtype=np.int32)
        self.france_cavalry = np.zeros(self.N, dtype=np.int32)
        self.france_artillery = np.zeros(self.N, dtype=np.int32)
        self.allies_infantry = np.zeros(self.N, dtype=np.int32)
        self.allies_cavalry = np.zeros(self.N, dtype=np.int32)
        self.allies_artillery = np.zeros(self.N, dtype=np.int32)

        self.occ_turns = np.zeros(self.N, dtype=np.int32)
        self.current_garrison = np.zeros(self.N, dtype=np.int32)
        self.sub_faction = np.full(self.N, SUBFACTION_NONE, dtype=np.int8)
        self.france_commander: List[Optional[str]] = [None] * self.N
        self.allies_commander: List[Optional[str]] = [None] * self.N

        self.france_army_history: Dict[int, deque] = {}
        self.allies_army_history: Dict[int, deque] = {}
        self.france_faction_history: deque = deque(maxlen=INITA_WINDOW)
        self.allies_faction_history: deque = deque(maxlen=INITA_WINDOW)

        self.france_available: List[str] = []
        self.allies_available: List[str] = []
        self.france_gone: set = set()
        self.allies_gone: set = set()

        # Reinforcement pools: men waiting at each depot node
        self.france_reinf_pool: Dict[str, int] = {n: 0 for n in FRANCE_REINF_DEPOTS}
        self.allies_reinf_pool: Dict[str, int] = {n: 0 for n in ALLIES_REINF_DEPOTS}

        self.turn = 0

    def _vprint(self, *args, **kwargs) -> None:
        """Print only when verbose=True (silenced during RL training)."""
        if self.verbose:
            print(*args, **kwargs)


    def reset(self, seed=None, options=None):
        """
        Main function. Goes through each process called per turn.
        """
        super().reset(seed=seed)

        if not hasattr(self, 'np_random') or self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # All nodes default to ALLIES
        # French army placements (below) will flip their nodes to FRANCE
        self.owner = np.full(self.N, ALLIES, dtype=np.int8)
        self.france_infantry = np.zeros(self.N, dtype=np.int32)
        self.france_cavalry = np.zeros(self.N, dtype=np.int32)
        self.france_artillery = np.zeros(self.N, dtype=np.int32)
        self.allies_infantry = np.zeros(self.N, dtype=np.int32)
        self.allies_cavalry = np.zeros(self.N, dtype=np.int32)
        self.allies_artillery = np.zeros(self.N, dtype=np.int32)
        self.occ_turns = np.zeros(self.N, dtype=np.int32)
        self.current_garrison = self._garrison_cap.copy()
        self.sub_faction = np.full(self.N, SUBFACTION_NONE, dtype=np.int8)
        self.france_commander = [None] * self.N
        self.allies_commander = [None] * self.N
        self.france_army_history = {}
        self.allies_army_history = {}
        self.france_faction_history = deque(maxlen=INITA_WINDOW)
        self.allies_faction_history = deque(maxlen=INITA_WINDOW)
        self.turn = 0

        # Reinforcement pools reset to zero
        self.france_reinf_pool = {n: 0 for n in FRANCE_REINF_DEPOTS}
        self.allies_reinf_pool = {n: 0 for n in ALLIES_REINF_DEPOTS}

        # Commander pools — remove all start-assigned commanders from the free pool
        # WELLINGTON is excluded at start; he arrives at turn 22 via historical event
        france_assigned = set(FRANCE_START_COMMANDERS.values()) - {'UNKNOWN'}
        self.france_available = [c for c in FRANCE_COMMANDER_POOL
                                  if c not in france_assigned]
        self.france_gone = set()

        allies_assigned = set(ALLIES_START_COMMANDERS.values()) - {'UNKNOWN'}
        _allies_not_yet = {'WELLINGTON'}
        self.allies_available = [c for c in ALLIES_COMMANDER_POOL
                                  if c not in allies_assigned and c not in _allies_not_yet]
        self.allies_gone = set()

        # Place starting armies
        # Per-node sizes from config.py
        for nid in FRANCE_START_NODES:
            if nid not in self.node_idx:
                continue
            i = self.node_idx[nid]
            size = FRANCE_START_SIZES.get(nid, INITIAL_ARMY_SIZE)
            sf = SUBFACTION_NONE
            inf, cav, arty = self._split_army(size, FRANCE, sf)
            self.france_infantry[i] = inf
            self.france_cavalry[i] = cav
            self.france_artillery[i] = arty
            self.owner[i] = FRANCE
            self.france_commander[i] = FRANCE_START_COMMANDERS.get(nid, 'UNKNOWN')
            self.current_garrison[i] = self._garrison_cap[i]

        for nid in ALLIES_START_NODES:
            if nid not in self.node_idx:
                continue
            i = self.node_idx[nid]
            size = ALLIES_START_SIZES.get(nid, INITIAL_ARMY_SIZE)
            sf = ALLIES_START_SUBFACTIONS.get(nid, SUBFACTION_NONE)
            inf, cav, arty = self._split_army(size, ALLIES, sf)
            self.allies_infantry[i] = inf
            self.allies_cavalry[i] = cav
            self.allies_artillery[i] = arty
            self.owner[i] = ALLIES
            self.sub_faction[i] = sf
            self.allies_commander[i] = ALLIES_START_COMMANDERS.get(nid, 'UNKNOWN')
            self.current_garrison[i] = self._garrison_cap[i]

        # Get observations for both factions
        obs = {'france': self._get_obs(FRANCE), 'allies': self._get_obs(ALLIES)}
        info = self._get_info()
        return obs, info

    def step(self, actions: Dict[str, np.ndarray]):
        assert self.owner is not None, "Call reset() before step()."

        # Snapshot ownership before any state changes (used by reward computation)
        prev_owner = self.owner.copy()

        # 0. Historical events
        self._apply_historical_events()

        # 1. Parse moves (before any state changes)
        f_moves = self._parse_all_moves(FRANCE, actions['france'])
        a_moves = self._parse_all_moves(ALLIES,  actions['allies'])

        # 2a. Resolve crossing (meeting-engagement) battles.
        # Retreats/advances are DEFERRED so that regular moves (2b) vacate nodes
        # first — preventing crossing-battle retreats from being wiped by a
        # departing friendly army that started from the same retreat destination.
        crossing_log, cross_f_srcs, cross_a_srcs, pending_crossing = \
            self._resolve_crossing_battles(f_moves, a_moves)

        # 2b. Apply non-crossing moves simultaneously
        f_moves_nc = [m for m in f_moves if m[0] not in cross_f_srcs]
        a_moves_nc = [m for m in a_moves if m[0] not in cross_a_srcs]
        self._apply_all_moves(FRANCE, f_moves_nc)
        self._apply_all_moves(ALLIES,  a_moves_nc)

        # 2c. Apply deferred crossing-battle retreats and advances.
        # Each entry is a 9-tuple snapshotted at battle resolution time so that
        # only the post-casualty amounts are moved — leaving any independent
        # friendly troops that arrived at the same source node via 2b untouched.
        for move in pending_crossing:
            faction, from_idx, to_idx, r_inf, r_cav, r_arty, r_cmd, r_sf, r_hist = move
            self._apply_deferred_move(
                faction, from_idx, to_idx,
                r_inf, r_cav, r_arty, r_cmd, r_sf, r_hist,
            )

        # 3. Resolve army-vs-army battles (non-crossing)
        battle_log = crossing_log + self._resolve_battles(f_moves_nc, a_moves_nc)

        # 4. Resolve sieges
        siege_log = self._resolve_sieges()

        # 5. Update ownership
        self._update_ownership()

        # 6. Advance occupation counters
        self._update_occ_turns()

        # 7. Reinforcements (pools accumulate; spawn or merge when threshold reached)
        self._step_reinforcements()

        self.turn += 1

        # 8. Termination checks
        france_alive = int(np.sum(self.france_infantry + self.france_cavalry + self.france_artillery)) > 0
        allies_alive = int(np.sum(self.allies_infantry + self.allies_cavalry + self.allies_artillery)) > 0
        dominance = self._check_dominance()
        terminated = (not france_alive or not allies_alive) or (dominance != NEUTRAL)
        truncated = self.turn >= MAX_TURNS

        obs = {'france': self._get_obs(FRANCE), 'allies': self._get_obs(ALLIES)}
        info = self._get_info()
        info['dominance_winner'] = dominance   # FRANCE / ALLIES

        # Reward computation
        france_r = self._compute_reward(prev_owner, battle_log, siege_log)
        allies_r = -france_r   # zero-sum for all competitive terms

        # Terminal reward: score difference
        # Applied on both termination (one side wiped or dominance) and truncation
        # (turn 313 reached) so the agent is always rewarded for the final
        # territorial situation, regardless of how the game ended.
        if terminated or truncated:
            score_diff = self._terminal_score()  # France - Allies
            france_r += score_diff
            allies_r -= score_diff

        # Depot penalty is non-zero-sum: each agent penalised only for its own
        france_r -= self._depot_penalty(FRANCE)
        allies_r -= self._depot_penalty(ALLIES)
        rewards = {'france': france_r, 'allies': allies_r}
        info['battles'] = battle_log
        info['sieges'] = siege_log
        # Per-turn movements (src -> dst node indices), exposed for the renderer
        # to animate smooth army motion. Only actual moves (src != dst) are listed.
        info['moves'] = (
            [('france', s, d, det) for (s, d, _i, _c, _a, det) in f_moves if s != d]
            + [('allies', s, d, det) for (s, d, _i, _c, _a, det) in a_moves if s != d]
        )

        if self.render_mode == 'human':
            self.render()

        return obs, rewards, terminated, truncated, info

    def render(self):
        """
        Display simulation information.
        """

        info = self._get_info()
        line = '─' * 66
        self._vprint(f"\n{line}")
        self._vprint(f"  Turn {self.turn:>3}/{MAX_TURNS}   "
              f"({'%.1f' % (self.turn / 52)} years in)")
        self._vprint(f"  France  — {info['france_troops']:>8,} troops   {info['france_nodes']:>3} nodes")
        self._vprint(f"  Allies  — {info['allies_troops']:>8,} troops   {info['allies_nodes']:>3} nodes")
        self._vprint(f"  Neutral — {info['neutral_nodes']:>3} nodes")
        self._vprint(f"\n  France armies:")
        for i in range(self.N):
            if self._has_army(FRANCE, i):
                cmd = self.france_commander[i] or 'UNKNOWN'
                post = self._post_label(i)
                self._vprint(f"    {self.node_ids[i]:4s}  "
                      f"inf={self.france_infantry[i]:>7,}  "
                      f"cav={self.france_cavalry[i]:>6,}  "
                      f"arty={self.france_artillery[i]:>4}  "
                      f"[{post}]  {cmd}")
        self._vprint(f"\n  Allied armies:")
        for i in range(self.N):
            if self._has_army(ALLIES, i):
                cmd = self.allies_commander[i] or 'UNKNOWN'
                sf = {SUBFACTION_BRITISH: 'BR', SUBFACTION_SPANISH: 'SP',
                        SUBFACTION_PORTUGUESE: 'PT'}.get(int(self.sub_faction[i]), '??')
                post = self._post_label(i)
                self._vprint(f"    {self.node_ids[i]:4s}  "
                      f"inf={self.allies_infantry[i]:>7,}  "
                      f"cav={self.allies_cavalry[i]:>6,}  "
                      f"arty={self.allies_artillery[i]:>4}  "
                      f"[{post}]  [{sf}]  {cmd}")

        # Reinforcement depot pools
        f_rate, a_rate = self._get_reinf_rate()
        self._vprint(f"\n  Reinforcement pools  (rate: FR {f_rate}/depot  AL {a_rate}/depot"
              f"  threshold: {REINF_SPAWN_THRESHOLD:,}):")
        for nid in FRANCE_REINF_DEPOTS:
            pool = self.france_reinf_pool.get(nid, 0)
            i = self.node_idx.get(nid, -1)
            blocked = (i >= 0 and self.owner[i] == ALLIES)
            pct = pool / REINF_SPAWN_THRESHOLD * 100
            tag = '  *** CAPTURED — no accumulation ***' if blocked else \
                      f'  ({pct:>5.1f}% of spawn threshold)'
            self._vprint(f"    FR depot {nid:4s}  pool={pool:>7,}{tag}")
        for nid in ALLIES_REINF_DEPOTS:
            pool = self.allies_reinf_pool.get(nid, 0)
            i = self.node_idx.get(nid, -1)
            blocked = (i >= 0 and self.owner[i] == FRANCE)
            pct = pool / REINF_SPAWN_THRESHOLD * 100
            tag = '  *** CAPTURED — no accumulation ***' if blocked else \
                      f'  ({pct:>5.1f}% of spawn threshold)'
            self._vprint(f"    AL depot {nid:4s}  pool={pool:>7,}{tag}")
        self._vprint(line)


    def _active_army_slots(self, faction: int) -> List[int]:
        """
        Returns a sorted list of node indices that currently hold an army
        for the given faction.  The position in this list is the slot index
        used by the per-army action_space (slot 0 = lowest node index, etc.).
        Length is always <= MAX_ARMIES.
        """
        return sorted(i for i in range(self.N) if self._has_army(faction, i))

    def legal_actions(self, faction: int) -> Dict[int, List[int]]:
        """
        Returns legal action integers keyed by SLOT INDEX (0-based position in
        the sorted active-army list, matching the per-army action_space).

            0               = stay
            1..MAX_DEGREE   = move all to neighbour d
            MAX_DEGREE+1..  = split 25% to neighbour d-MAX_DEGREE
                              (only offered when faction is below MAX_ARMIES cap)

        Use _active_army_slots(faction)[slot] to recover the node index for a slot.
        """
        active = self._active_army_slots(faction)
        can_split = len(active) < MAX_ARMIES
        result = {}
        for slot, i in enumerate(active):
            # An army is only allowed to split if it has at least
            # 2*MIN_ARMY_SIZE men
            split_ok = can_split and self._men(faction, i) >= 2 * MIN_ARMY_SIZE
            dirs = [0]
            for d, nbr in enumerate(self.neighbour_list[self.node_ids[i]], start=1):
                if nbr is not None:
                    dirs.append(d)
                    if split_ok:
                        dirs.append(d + MAX_DEGREE)
            result[slot] = dirs
        return result



    def _has_army(self, faction: int, i: int) -> bool:
        if faction == FRANCE:
            return int(self.france_infantry[i] + self.france_cavalry[i] + self.france_artillery[i]) > 0
        return int(self.allies_infantry[i] + self.allies_cavalry[i] + self.allies_artillery[i]) > 0

    def _men(self, faction: int, i: int) -> int:
        """Infantry + cavalry (men count) for one faction at one node."""
        if faction == FRANCE:
            return int(self.france_infantry[i] + self.france_cavalry[i])
        return int(self.allies_infantry[i] + self.allies_cavalry[i])

    def _split_army(self, total: int, faction: int, subfaction: int) -> Tuple[int, int, int]:
        """
        Split a total headcount into (infantry, cavalry, artillery_guns)
        using the faction/subfaction ratios from config.
        """
        key = FRANCE if faction == FRANCE else subfaction
        cav_ratio = CAVALRY_RATIO.get(key, CAVALRY_RATIO[FRANCE])
        arty_rate  = ARTY_PER_1000.get(key, ARTY_PER_1000[FRANCE])
        cav  = round(total * cav_ratio)
        arty = round(total / 1000 * arty_rate)
        inf  = total - cav
        return inf, cav, arty

    def _apply_casualties(self, faction: int, i: int, men_cas: int, arty_cas: int):
        """
        Apply post-battle casualties in-place.
        men_cas split: CAVALRY_CASUALTY_RATIO → cavalry, remainder → infantry.
        arty_cas applied directly to guns.
        """
        cav_cas = round(men_cas * CAVALRY_CASUALTY_RATIO)
        inf_cas = men_cas - cav_cas
        if faction == FRANCE:
            self.france_cavalry[i] = max(0, int(self.france_cavalry[i])   - cav_cas)
            self.france_infantry[i] = max(0, int(self.france_infantry[i])  - inf_cas)
            self.france_artillery[i] = max(0, int(self.france_artillery[i]) - arty_cas)
        else:
            self.allies_cavalry[i] = max(0, int(self.allies_cavalry[i])   - cav_cas)
            self.allies_infantry[i] = max(0, int(self.allies_infantry[i])  - inf_cas)
            self.allies_artillery[i] = max(0, int(self.allies_artillery[i]) - arty_cas)

    def _march_attrition(
        self, inf: int, cav: int, arty: int, src: int, dst: int
    ) -> Tuple[int, int, int]:
        """
        Apply road-march attrition to troops moving from src to dst.

        Rules (from config):
            Primary -> no penalty for any army.
            Secondary -> armies with men > MARCH_SECONDARY_THRESHOLD lose
                        MARCH_SECONDARY_RATE of their men.
            Tertiary -> armies with men > MARCH_TERTIARY_THRESHOLD lose
                        MARCH_TERTIARY_RATE of their men.

        Artillery is unaffected.  Men losses are split infantry/cavalry
        using the standard CAVALRY_CASUALTY_RATIO.
        Stay moves (src == dst) are always exempt.
        """

        # Stay are exempt
        if src == dst:
            return inf, cav, arty

        src_id = self.node_ids[src]
        dst_id = self.node_ids[dst]
        road_type = self._road_type.get((src_id, dst_id), 'primary')
        men = inf + cav

        rate = 0.0
        if road_type == 'tertiary' and men > MARCH_TERTIARY_THRESHOLD:
            rate = MARCH_TERTIARY_RATE
        elif road_type == 'secondary' and men > MARCH_SECONDARY_THRESHOLD:
            rate = MARCH_SECONDARY_RATE

        if rate == 0.0:
            return inf, cav, arty

        men_loss = round(men * rate)
        cav_loss = round(men_loss * CAVALRY_CASUALTY_RATIO)
        inf_loss = men_loss - cav_loss
        return max(0, inf - inf_loss), max(0, cav - cav_loss), arty

    def _wipe_army(self, faction: int, i: int):
        """
        Zero all unit arrays for one faction at one node.
        """
        if faction == FRANCE:
            self.france_infantry[i] = 0
            self.france_cavalry[i] = 0
            self.france_artillery[i] = 0
        else:
            self.allies_infantry[i] = 0
            self.allies_cavalry[i] = 0
            self.allies_artillery[i] = 0

    def _eliminate_remnant(self, faction: int, i: int, cause: str = '') -> bool:
        """
        If a faction's army at node i is a NONZERO remnant below MIN_ARMY_SIZE
        men, wipe it from the map and return its commander to the available pool.

        Returns True if an army was eliminated. Used wherever attrition (battle,
        siege, march) can leave a force too small to be worth keeping.
        """
        men = self._men(faction, i)
        if not (0 < men < MIN_ARMY_SIZE):
            return False
        cmd_arr = self.france_commander if faction == FRANCE else self.allies_commander
        cur = cmd_arr[i]
        if cur and cur != 'UNKNOWN':
            self._release_commander(faction, cur)
        self._wipe_army(faction, i)
        cmd_arr[i] = None
        self._vprint(f"  {'France' if faction == FRANCE else 'Allies'} army at "
                     f"{self.node_ids[i]} eliminated "
                     f"({men} < {MIN_ARMY_SIZE} men{', ' + cause if cause else ''})")
        return True

    def _find_retreat_node(
        self,
        battle_node_idx: int,
        loser: int,
        loser_is_attacker: bool,
        att_src_idx: Optional[int],
    ) -> Optional[int]:
        """
        Return the index of the node the losing army should retreat to.

        Attacker loser (and att_src is known) → retreat to att_src_idx.
        Defender loser (or attacker with no source) → search neighbours:
            1. Closest own-faction neighbour
            2. Closest neutral neighbour
            3. Closest any other neighbour  (excluding att_src)
        Returns None only if the battle node is completely isolated.
        """
        if loser_is_attacker and att_src_idx is not None:
            return att_src_idx

        nid = self.node_ids[battle_node_idx]
        nbrs: List[Tuple[float, int]] = []
        for nbr_name, dist in zip(self.neighbour_list[nid], self.neighbour_dist[nid]):
            if nbr_name is None:
                continue
            nbrs.append((dist if dist is not None else 1.0, self.node_idx[nbr_name]))
        nbrs.sort()

        if not nbrs:
            return None  # isolated node, army destroyed

        # Prefer nodes that are not the attacker's source
        non_src = [(d, idx) for d, idx in nbrs if idx != att_src_idx]
        candidates = non_src if non_src else nbrs

        # Priority 1: closest own-faction node
        for _, idx in candidates:
            if self.owner[idx] == loser:
                return idx
        # Priority 2: closest neutral node
        for _, idx in candidates:
            if self.owner[idx] == NEUTRAL:
                return idx
        # Priority 3: any closest node (even enemy-owned)
        return candidates[0][1]

    def _retreat_army(self, faction: int, from_idx: int, to_idx: int):
        """
        Move all troops of a faction from from_idx to to_idx, merging with
        any existing forces there.  The largest army (by men) wins metadata
        (commander, sub_faction, history).
        """
        if faction == FRANCE:
            inf_arr  = self.france_infantry
            cav_arr  = self.france_cavalry
            arty_arr = self.france_artillery
            cmd_arr  = self.france_commander
            hist     = self.france_army_history
        else:
            inf_arr  = self.allies_infantry
            cav_arr  = self.allies_cavalry
            arty_arr = self.allies_artillery
            cmd_arr  = self.allies_commander
            hist     = self.allies_army_history

        # Snapshot retreating army
        r_inf = int(inf_arr[from_idx])
        r_cav = int(cav_arr[from_idx])
        r_arty = int(arty_arr[from_idx])
        r_cmd = cmd_arr[from_idx]
        r_sf = int(self.sub_faction[from_idx]) if faction == ALLIES else SUBFACTION_NONE
        r_hist = list(hist.get(from_idx, []))
        r_men = r_inf + r_cav

        existing_men = int(inf_arr[to_idx]) + int(cav_arr[to_idx])

        # Clear source
        inf_arr[from_idx] = 0
        cav_arr[from_idx] = 0
        arty_arr[from_idx] = 0
        cmd_arr[from_idx] = None
        if faction == ALLIES:
            self.sub_faction[from_idx] = SUBFACTION_NONE
        hist.pop(from_idx, None)

        # Merge into destination
        inf_arr[to_idx] += r_inf
        cav_arr[to_idx] += r_cav
        arty_arr[to_idx] += r_arty

        existing_cmd = cmd_arr[to_idx]

        # Commander: most senior leads the merged force
        r_sen = COMMANDER_SENIORITY.get((r_cmd or '').upper(),
                                         COMMANDER_SENIORITY_DEFAULT)
        e_sen = COMMANDER_SENIORITY.get((existing_cmd or '').upper(),
                                         COMMANDER_SENIORITY_DEFAULT)
        if r_cmd is not None and r_sen <= e_sen:
            # Retreating commander is more senior, they take command
            self._release_commander(faction, existing_cmd)
            cmd_arr[to_idx] = r_cmd
        else:
            # Existing commander is more senior, release the retreating one
            self._release_commander(faction, r_cmd)

        # Subfaction / history: largest army's values dominate
        if r_men >= existing_men:
            if faction == ALLIES:
                self.sub_faction[to_idx] = r_sf
            if r_hist:
                hist[to_idx] = deque(r_hist, maxlen=MOMNTA_WINDOW)

    def _apply_deferred_move(
        self,
        faction: int,
        from_idx: int,
        to_idx: int,
        r_inf: int,
        r_cav: int,
        r_arty: int,
        r_cmd: Optional[str],
        r_sf: int,
        r_hist: list,
    ):
        """
        Apply a snapshotted crossing-battle retreat or advance.

        Subtracts exactly (r_inf, r_cav, r_arty) from from_idx, leaving any
        additional troops that arrived there independently via _apply_all_moves,
        and merges them into to_idx.

        This is the key difference from _retreat_army: rather than moving ALL
        troops from the source, we only move the amounts captured at battle
        resolution time.  This prevents a deferred retreat from accidentally
        absorbing a friendly army that moved into the same node afterwards.

        Metadata (commander, sub_faction, history) is resolved at the
        destination using the largest-army-wins rule.  The source is cleaned
        only when it ends up completely empty.
        """
        if r_inf + r_cav + r_arty == 0:
            return   # Army was completely destroyed by casualties, nothing to move

        if faction == FRANCE:
            inf_arr = self.france_infantry
            cav_arr = self.france_cavalry
            arty_arr = self.france_artillery
            cmd_arr = self.france_commander
            hist = self.france_army_history
        else:
            inf_arr = self.allies_infantry
            cav_arr = self.allies_cavalry
            arty_arr = self.allies_artillery
            cmd_arr = self.allies_commander
            hist = self.allies_army_history

        # Move at most what the snapshot asked for (just for safety)
        move_inf = min(r_inf,  int(inf_arr[from_idx]))
        move_cav = min(r_cav,  int(cav_arr[from_idx]))
        move_arty = min(r_arty, int(arty_arr[from_idx]))
        move_men = move_inf + move_cav

        # Subtract snapshotted amounts from source
        inf_arr[from_idx] -= move_inf
        cav_arr[from_idx] -= move_cav
        arty_arr[from_idx] -= move_arty

        # If source is now empty, clean up its metadata
        if not self._has_army(faction, from_idx):
            cmd_arr[from_idx] = None
            if faction == ALLIES:
                self.sub_faction[from_idx] = SUBFACTION_NONE
            hist.pop(from_idx, None)
        # If source still has troops (another army arrived), leave its metadata untouched

        # Merge into destination
        existing_men = int(inf_arr[to_idx]) + int(cav_arr[to_idx])
        existing_cmd = cmd_arr[to_idx]

        inf_arr[to_idx] += move_inf
        cav_arr[to_idx] += move_cav
        arty_arr[to_idx] += move_arty

        # Commander: most senior leads the merged force
        r_sen = COMMANDER_SENIORITY.get((r_cmd or '').upper(),
                                         COMMANDER_SENIORITY_DEFAULT)
        e_sen = COMMANDER_SENIORITY.get((existing_cmd or '').upper(),
                                         COMMANDER_SENIORITY_DEFAULT)
        if r_cmd is not None and r_sen <= e_sen:
            self._release_commander(faction, existing_cmd)
            cmd_arr[to_idx] = r_cmd
        else:
            self._release_commander(faction, r_cmd)

        # Subfaction / history: largest army's values dominate
        if move_men >= existing_men:
            if faction == ALLIES:
                self.sub_faction[to_idx] = r_sf
            if r_hist:
                hist[to_idx] = deque(r_hist, maxlen=MOMNTA_WINDOW)



    def _count_armies(self, faction: int) -> int:
        return sum(1 for i in range(self.N) if self._has_army(faction, i))

    def _parse_all_moves(
        self, faction: int, action: np.ndarray
    ) -> List[Tuple[int, int, int, int, int, bool]]:
        """
        Parse actions into move tuples: (src, dst, inf, cav, arty, is_detachment).

        Action encoding per node:
            0 -> stay (all troops)
            1 .. MAX_DEGREE -> move ALL troops to neighbour d
            MAX_DEGREE+1 .. 2*MAX_DEGREE -> split: send 25% to neighbour d-MAX_DEGREE,
                                           keep 75% in place.Blocked (treated as stay) 
                                           if faction is already at MAX_ARMIES armies.
        """
        active = self._active_army_slots(faction)
        army_count = len(active)
        moves: List[Tuple[int, int, int, int, int, bool]] = []

        if faction == FRANCE:
            inf_arr = self.france_infantry
            cav_arr = self.france_cavalry
            arty_arr = self.france_artillery
        else:
            inf_arr = self.allies_infantry
            cav_arr = self.allies_cavalry
            arty_arr = self.allies_artillery

        for slot, i in enumerate(active):
            direction = int(action[slot])
            inf = int(inf_arr[i])
            cav = int(cav_arr[i])
            arty = int(arty_arr[i])
            nid = self.node_ids[i]

            if direction == 0:
                # Stay — all troops remain
                moves.append((i, i, inf, cav, arty, False))
                continue

            is_split = direction > MAX_DEGREE
            d = direction - MAX_DEGREE if is_split else direction
            nbr = self.neighbour_list[nid][d - 1] if d - 1 < len(self.neighbour_list[nid]) else None

            if nbr is None:
                # Invalid direction, stay
                moves.append((i, i, inf, cav, arty, False))
                continue

            dst = self.node_idx[nbr]

            if is_split:
                if army_count >= MAX_ARMIES:
                    # Cap reached, treat as move-all instead of split
                    moves.append((i, dst, inf, cav, arty, False))
                else:
                    # 25% detachment goes forward, 75% main body stays
                    det_inf = inf  // 4
                    det_cav = cav  // 4
                    det_arty = arty // 4
                    main_inf = inf  - det_inf
                    main_cav = cav  - det_cav
                    main_arty = arty - det_arty
                    moves.append((i, dst, det_inf,  det_cav,  det_arty,  True))   # detachment
                    moves.append((i, i,   main_inf, main_cav, main_arty, False))  # main body stays
                    army_count += 1   # track locally so successive splits also respect the cap
            else:
                # Move all
                moves.append((i, dst, inf, cav, arty, False))

        return moves

    def _apply_all_moves(self, faction: int, moves: List[Tuple[int, int, int, int, int, bool]]):
        """
        Two-phase simultaneous move.  Propagates all six unit arrays plus
        commander / sub_faction / battle-history.

        Move tuple: (src, dst, inf, cav, arty, is_detachment)
          is_detachment=True -> 25% split detachment; gets a fresh commander from pool.
          is_detachment=False -> full army or main body; largest non-detachment wins metadata.
        """
        if faction == FRANCE:
            inf_arr = self.france_infantry
            cav_arr = self.france_cavalry
            arty_arr = self.france_artillery
            cmd_arr = self.france_commander
            hist = self.france_army_history
        else:
            inf_arr = self.allies_infantry
            cav_arr = self.allies_cavalry
            arty_arr = self.allies_artillery
            cmd_arr = self.allies_commander
            hist = self.allies_army_history

        # Snapshot metadata before clearing (source positions)
        src_meta: Dict[int, Tuple[str, int, list]] = {}
        for src, dst, inf, cav, arty, is_det in moves:
            if src not in src_meta:
                src_meta[src] = (
                    cmd_arr[src],
                    int(self.sub_faction[src]) if faction == ALLIES else SUBFACTION_NONE,
                    list(hist.get(src, [])),
                )

        # Phase 1: clear all source nodes
        cleared: set = set()
        for src, dst, inf, cav, arty, is_det in moves:
            if src not in cleared:
                inf_arr[src] = 0
                cav_arr[src] = 0
                arty_arr[src] = 0
                if faction == ALLIES:
                    self.sub_faction[src] = SUBFACTION_NONE
                cmd_arr[src] = None
                hist.pop(src, None)
                cleared.add(src)

        # Phase 2: accumulate troops at destinations.
        #
        # Commander selection  — SENIORITY: the most senior commander seen 
        # across all armies merging at a destination leads the combined force.  
        # Detachments contribute troops but NOT their source commander.
        #
        # Subfaction / history — LARGEST ARMY: the non-detachment with the
        # most men determines these values.  A detachment destination that
        # receives no non-detachment armies gets fresh metadata.
        #
        # best[dst] = (best_cmd, best_seniority,   # commander track
        #              largest_nd_men, nd_sf, nd_hist,  # metadata track
        #              has_non_det)                 # flag: any non-det arrived?
        best: Dict[int, Tuple] = {}
        attrition_hit: set = set()   # destinations that actually lost men en route

        for src, dst, inf, cav, arty, is_det in moves:
            # Road-march attrition: large armies on secondary/tertiary roads
            # lose a fraction of men before arriving at the destination.
            men_pre = inf + cav
            inf, cav, arty = self._march_attrition(inf, cav, arty, src, dst)
            if inf + cav < men_pre:
                attrition_hit.add(dst)

            inf_arr[dst] += inf
            cav_arr[dst] += cav
            arty_arr[dst] += arty
            men = inf + cav
            cmd, sf, h = src_meta[src]

            if is_det:
                # Detachment: contributes troops only; commander handled later
                if dst not in best:
                    best[dst] = (None, COMMANDER_SENIORITY_DEFAULT,
                                 0, sf, h, False)
                continue

            # Non-detachment
            cmd_sen = COMMANDER_SENIORITY.get((cmd or '').upper(),
                                               COMMANDER_SENIORITY_DEFAULT)
            if dst not in best:
                best[dst] = (cmd, cmd_sen, men, sf, h, True)
            else:
                prev_cmd, prev_sen, prev_men, prev_sf, prev_h, prev_nd = best[dst]

                # Commander: most senior wins
                if cmd_sen < prev_sen:
                    self._release_commander(faction, prev_cmd)
                    win_cmd, win_sen = cmd, cmd_sen
                else:
                    self._release_commander(faction, cmd)
                    win_cmd, win_sen = prev_cmd, prev_sen

                # Subfaction / history: largest non-det wins
                if men > prev_men:
                    win_men, win_sf, win_h = men, sf, h
                else:
                    win_men, win_sf, win_h = prev_men, prev_sf, prev_h

                best[dst] = (win_cmd, win_sen, win_men, win_sf, win_h, True)

        # Apply metadata to each destination.
        # Detachment-only destinations get a fresh pool commander.
        for dst, (best_cmd, _, nd_men, nd_sf, nd_hist, has_nd) in best.items():
            if has_nd:
                cmd_arr[dst] = best_cmd if best_cmd is not None \
                               else self._pick_commander(faction)
                if faction == ALLIES:
                    self.sub_faction[dst] = nd_sf
                if nd_hist:
                    hist[dst] = deque(nd_hist, maxlen=MOMNTA_WINDOW)
            else:
                # Only detachments arrived —> assign most senior available
                cmd_arr[dst] = self._pick_commander(faction)
                # Detachments inherit their source army's subfaction
                # Battle history not carried over
                if faction == ALLIES:
                    self.sub_faction[dst] = nd_sf

        # Any resulting force below MIN_ARMY_SIZE is eliminated, whether it was
        # ground down by road-march attrition, or simply started out too
        # small (like a split detachment).
        for dst in best:
            cause = 'march attrition' if dst in attrition_hit else 'undersized remnant'
            self._eliminate_remnant(faction, dst, cause=cause)

    def _supply_distance(self, node_idx: int, faction: int) -> float:
        """
        Dijkstra from node_idx to the nearest supply depot owned by `faction`,
        travelling only through nodes NOT owned by the enemy faction.
        Returns distance in km, or 999_999 if no path exists.

        Supply depots are considered cities and above.
        """
        enemy = ALLIES if faction == FRANCE else FRANCE

        if node_idx in self._supply_depots and self.owner[node_idx] == faction:
            return 0.0

        dist_km: Dict[int, float] = {node_idx: 0.0}
        heap = [(0.0, node_idx)]

        while heap:
            cost, cur = heapq.heappop(heap)
            if cost > dist_km.get(cur, float('inf')):
                continue
            nid = self.node_ids[cur]
            for nbr_name, edge_km in zip(self.neighbour_list[nid], self.neighbour_dist[nid]):
                if nbr_name is None:
                    continue
                nbr_idx = self.node_idx[nbr_name]
                if self.owner[nbr_idx] == enemy:
                    continue
                new_cost = cost + edge_km
                if new_cost < dist_km.get(nbr_idx, float('inf')):
                    dist_km[nbr_idx] = new_cost
                    if nbr_idx in self._supply_depots and self.owner[nbr_idx] == faction:
                        return new_cost
                    heapq.heappush(heap, (new_cost, nbr_idx))

        return 999_999.0

    def _resolve_crossing_battles(
        self,
        f_moves: List[Tuple[int, int, int]],
        a_moves: List[Tuple[int, int, int]],
    ) -> Tuple[List[dict], set, set, List[Tuple]]:
        """
        Called BEFORE _apply_all_moves.  Detects pairs of moves that traverse
        the same edge in opposite directions (France A->B, Allies B->A) and
        resolves them as meeting engagements.

        Rules
        -----
        - Attacker  = army with fewer men (moves faster, meets first).
        - Battle node = defender's source node.
        - occ_turns forced to 0 (HD) — defender had already left.
        - Retreat anchor = loser's origin node (not the battle node).
        - If the defender wins, they proceed to their intended destination.

        Retreats and advances are NOT applied here — they are returned as
        pending_moves (list of (faction, from_idx, to_idx)) so that step()
        can apply them AFTER _apply_all_moves.  This prevents a regular move
        vacating a node from wiping crossing-battle retreat troops that were
        deposited there first.

        Returns
        log: list of battle dicts (same schema as _resolve_battles)
        cross_f_srcs: France source-node indices involved
        cross_a_srcs: Allies source-node indices involved
        pending_moves: deferred (faction, from_idx, to_idx) retreat/advance pairs
        """
        # Build lookup for allied moves: (src, dst) → (src, dst)
        # Only "move all" (is_det=False, src!=dst) moves can trigger a crossing.
        # Detachments are small flanking forces; they don't constitute a full crossing.
        a_by_edge: Dict[Tuple[int, int], Tuple[int, int]] = {
            (src, dst): (src, dst)
            for src, dst, _inf, _cav, _arty, is_det in a_moves
            if src != dst and not is_det
        }

        crossing_pairs: List[Tuple[int, int, int, int]] = []  # (f_src, f_dst, a_src, a_dst)
        cross_f_srcs: set = set()
        cross_a_srcs: set = set()

        for src, dst, _inf, _cav, _arty, is_det in f_moves:
            if src == dst or is_det:
                continue
            rev = (dst, src)
            if rev in a_by_edge:
                a_src, a_dst = a_by_edge[rev]
                crossing_pairs.append((src, dst, a_src, a_dst))
                cross_f_srcs.add(src)
                cross_a_srcs.add(a_src)

        if not crossing_pairs:
            return [], cross_f_srcs, cross_a_srcs, []

        log = []
        pending_moves: List[Tuple] = []
        for f_src, f_dst, a_src, a_dst in crossing_pairs:
            # Troops are still at their source positions
            f_inf = int(self.france_infantry[f_src])
            f_cav = int(self.france_cavalry[f_src])
            f_arty = int(self.france_artillery[f_src])
            a_inf = int(self.allies_infantry[a_src])
            a_cav = int(self.allies_cavalry[a_src])
            a_arty = int(self.allies_artillery[a_src])

            f_men = f_inf + f_cav
            a_men = a_inf + a_cav

            # Attacker = fewer men; battle at defender's source node
            if f_men <= a_men:
                attacker = FRANCE
                att_src_idx = f_src
                def_src_idx = a_src   # battle node
                def_dst_idx = a_dst   # where defender was heading
            else:
                attacker = ALLIES
                att_src_idx = a_src
                def_src_idx = f_src
                def_dst_idx = f_dst 

            battle_idx = def_src_idx
            nid = self.node_ids[battle_idx]

            # Sub-factions (based on each army's source position)
            att_sf = SUBFACTION_NONE if attacker == FRANCE else int(self.sub_faction[a_src])
            def_sf = int(self.sub_faction[a_src]) if attacker == FRANCE else SUBFACTION_NONE

            # Commanders
            att_cmd = (self.france_commander[f_src] or 'UNKNOWN') if attacker == FRANCE \
                      else (self.allies_commander[a_src] or 'UNKNOWN')
            def_cmd = (self.allies_commander[a_src] or 'UNKNOWN') if attacker == FRANCE \
                      else (self.france_commander[f_src] or 'UNKNOWN')

            # Army histories
            att_hist = list(self.france_army_history.get(f_src, []) if attacker == FRANCE
                             else self.allies_army_history.get(a_src, []))
            def_hist = list(self.allies_army_history.get(a_src, []) if attacker == FRANCE
                             else self.france_army_history.get(f_src, []))
            att_wins = sum(att_hist); att_total = len(att_hist)
            def_wins = sum(def_hist); def_total = len(def_hist)

            # Faction histories
            f_fact = list(self.france_faction_history)
            a_fact = list(self.allies_faction_history)
            att_fw = sum(f_fact if attacker == FRANCE else a_fact)
            att_ft = len(f_fact if attacker == FRANCE else a_fact)
            def_fw = sum(a_fact if attacker == FRANCE else f_fact)
            def_ft = len(a_fact if attacker == FRANCE else f_fact)

            # Supply distances (anchored to each army's current/source position)
            att_sup = self._supply_distance(att_src_idx, attacker)
            def_sup = self._supply_distance(def_src_idx, ALLIES if attacker == FRANCE else FRANCE)

            # Road type — always 'primary' in crossing battles: both armies are
            # in transit so neither side can claim a surprise advantage (surpa=0).
            road_type = 'primary'

            winner, f_men_cas, a_men_cas, f_arty_cas, a_arty_cas = resolve_battle(
                france_infantry = f_inf,
                france_cavalry = f_cav,
                france_artillery = f_arty,
                allies_infantry = a_inf,
                allies_cavalry = a_cav,
                allies_artillery = a_arty,
                terra1 = self._terra1[nid],
                terra2 = self._terra2[nid],
                occ_turns = 0,           # HD — defender had already left
                attacker_faction = attacker,
                attacker_subfaction = att_sf,
                defender_subfaction = def_sf,
                att_commander = att_cmd,
                def_commander = def_cmd,
                road_type = road_type,
                att_supply_dist = att_sup,
                def_supply_dist = def_sup,
                att_army_wins = att_wins,
                att_army_total = att_total,
                def_army_wins = def_wins,
                def_army_total = def_total,
                att_faction_wins = att_fw,
                att_faction_total = att_ft,
                def_faction_wins = def_fw,
                def_faction_total = def_ft,
            )

            # Verbose debug print
            _att_q = COMMANDER_QUALITY.get(att_cmd.upper(), COMMANDER_QUALITY_DEFAULT)
            _def_q = COMMANDER_QUALITY.get(def_cmd.upper(), COMMANDER_QUALITY_DEFAULT)
            _sf_name = {-1: 'French', 0: 'British', 1: 'Spanish', 2: 'Portuguese'}
            self._vprint(
                f"\n{'='*62}\n"
                f"  CROSSING BATTLE at {nid}  (turn {self.turn})\n"
                f"  (meeting engagement — both armies were in transit)\n"
                f"{'='*62}\n"
                f"  Attacker : {'France' if attacker == FRANCE else 'Allies'}"
                f"  [{_sf_name.get(att_sf, '?')}]  cmd={att_cmd} (q={_att_q:.3f})\n"
                f"  Defender : {'France' if attacker != FRANCE else 'Allies'}"
                f"  [{_sf_name.get(def_sf, '?')}]  cmd={def_cmd} (q={_def_q:.3f})\n"
                f"\n── Pre-battle armies ──────────────────────────────────────\n"
                f"  France   inf={f_inf:6d}  cav={f_cav:5d}  arty={f_arty:4d}\n"
                f"  Allies   inf={a_inf:6d}  cav={a_cav:5d}  arty={a_arty:4d}\n"
                f"\n── Model outputs ──────────────────────────────────────────\n"
                f"  WINNER : {'France' if winner == FRANCE else 'Allies'}\n"
                f"  France casualties : men={f_men_cas:5d}  arty={f_arty_cas:3d}\n"
                f"  Allies casualties : men={a_men_cas:5d}  arty={a_arty_cas:3d}\n"
                f"{'='*62}\n"
            )

            # Apply casualties at source positions
            self._apply_casualties(FRANCE, f_src, f_men_cas, f_arty_cas)
            self._apply_casualties(ALLIES, a_src, a_men_cas, a_arty_cas)

            # Winner captures loser's lost guns
            if winner == FRANCE:
                self.france_artillery[f_src] += a_arty_cas
            else:
                self.allies_artillery[a_src] += f_arty_cas

            # Commander post-battle resolution
            f_cmd_died = False
            a_cmd_died = False

            winner_src = f_src if winner == FRANCE else a_src
            loser_src = a_src if winner == FRANCE else f_src
            loser = ALLIES if winner == FRANCE else FRANCE

            winner_cmd_arr = self.france_commander if winner == FRANCE else self.allies_commander
            loser_cmd_arr = self.allies_commander if winner == FRANCE else self.france_commander
            winner_gone = self.france_gone if winner == FRANCE else self.allies_gone
            loser_gone = self.allies_gone if winner == FRANCE else self.france_gone

            loser_cmd = loser_cmd_arr[loser_src]
            loser_cmd_alive = True
            if loser_cmd and loser_cmd != 'UNKNOWN':
                if self.np_random.random() < COMMANDER_DEATH_PROB:
                    loser_gone.add(loser_cmd)
                    loser_cmd_alive = False
                    replacement = self._pick_commander(loser)
                    loser_cmd_arr[loser_src] = replacement
                    if winner == FRANCE: a_cmd_died = True
                    else:                f_cmd_died = True
                    loser_faction = 'France' if loser == FRANCE else 'Allies'
                    self._vprint(f"  [KIA] {loser_cmd} ({loser_faction}) killed in action"
                          + (f" — replaced by {replacement}" if replacement != 'UNKNOWN'
                             else " — no replacement available"))

            winner_cmd = winner_cmd_arr[winner_src]
            if winner_cmd and winner_cmd != 'UNKNOWN':
                if self.np_random.random() < COMMANDER_DEATH_PROB:
                    winner_gone.add(winner_cmd)
                    replacement = self._pick_commander(winner)
                    winner_cmd_arr[winner_src] = replacement
                    if winner == FRANCE: f_cmd_died = True
                    else:                a_cmd_died = True
                    winner_faction = 'France' if winner == FRANCE else 'Allies'
                    self._vprint(f"  [KIA] {winner_cmd} ({winner_faction}) killed in action"
                          + (f" — replaced by {replacement}" if replacement != 'UNKNOWN'
                             else " — no replacement available"))

            # Battle histories
            # Store win/loss at the armies' current source nodes
            france_won = (winner == FRANCE)
            self.france_faction_history.append(1 if france_won else 0)
            self.allies_faction_history.append(0 if france_won else 1)
            if winner == FRANCE:
                self.france_army_history.setdefault(winner_src, deque(maxlen=MOMNTA_WINDOW)).append(1)
                self.allies_army_history.setdefault(loser_src,  deque(maxlen=MOMNTA_WINDOW)).append(0)
            else:
                self.allies_army_history.setdefault(winner_src, deque(maxlen=MOMNTA_WINDOW)).append(1)
                self.france_army_history.setdefault(loser_src,  deque(maxlen=MOMNTA_WINDOW)).append(0)

            # Loser retreats to a neighbour of their own source node
            # The battle was midway — the loser never reached their destination
            loser_name = 'France' if loser == FRANCE else 'Allies'
            retreat_node = None
            if not self._has_army(loser, loser_src):
                self._vprint(f"  Retreat  : {loser_name} DESTROYED by casualties")
                if loser_cmd_alive and loser_cmd and loser_cmd != 'UNKNOWN':
                    self._release_commander(loser, loser_cmd)
                loser_cmd_arr[loser_src] = None
            elif self._men(loser, loser_src) < MIN_ARMY_SIZE:

                # Remnant too small to remain a fighting force — eliminate it and
                # return its commander to the pool
                self._vprint(f"  Retreat  : {loser_name} REMNANT "
                             f"({self._men(loser, loser_src)} < {MIN_ARMY_SIZE} men) — eliminated")
                cur_cmd = loser_cmd_arr[loser_src]
                if cur_cmd and cur_cmd != 'UNKNOWN':
                    self._release_commander(loser, cur_cmd)
                self._wipe_army(loser, loser_src)
                loser_cmd_arr[loser_src] = None
            else:
                # Snapshot loser state at this moment
                _l_inf_arr = self.allies_infantry  if loser == ALLIES else self.france_infantry
                _l_cav_arr = self.allies_cavalry   if loser == ALLIES else self.france_cavalry
                _l_arty_arr = self.allies_artillery if loser == ALLIES else self.france_artillery
                _l_hist_map = self.allies_army_history if loser == ALLIES else self.france_army_history
                snap_l_inf = int(_l_inf_arr[loser_src])
                snap_l_cav = int(_l_cav_arr[loser_src])
                snap_l_arty = int(_l_arty_arr[loser_src])
                snap_l_cmd = loser_cmd_arr[loser_src] 
                snap_l_sf = int(self.sub_faction[loser_src]) if loser == ALLIES else SUBFACTION_NONE
                snap_l_hist = list(_l_hist_map.get(loser_src, []))

                retreat_to = self._find_retreat_node(
                    loser_src,   # search neighbours of the loser's origin
                    loser,
                    False,       # never shortcut to origin — always search neighbours
                    battle_idx,  # exclude: winner is advancing from/through here
                )
                if retreat_to is not None:
                    # Defer retreat (9-tuple) until after _apply_all_moves
                    pending_moves.append((loser, loser_src, retreat_to,
                                          snap_l_inf, snap_l_cav, snap_l_arty,
                                          snap_l_cmd, snap_l_sf, snap_l_hist))
                    retreat_node = self.node_ids[retreat_to]
                    self._vprint(f"  Retreat  : {loser_name} → {retreat_node}")
                else:
                    # If no retreat is possible, eliminate the army
                    self._vprint(f"  Retreat  : {loser_name} SURROUNDED — disbanded")
                    if loser_cmd_alive and loser_cmd and loser_cmd != 'UNKNOWN':
                        self._release_commander(loser, loser_cmd)
                    self._wipe_army(loser, loser_src)
                    loser_cmd_arr[loser_src] = None

            # Winner proceeds to their intended destination
            # Attacker wins -> advances to att_dst_idx (= battle_idx = def_src)
            # Defender wins -> advances to def_dst_idx (= att_src)
            #
            # SNAPSHOT winner state NOW for the same reason as the loser retreat.
            defender = ALLIES if attacker == FRANCE else FRANCE
            winner_dst = def_dst_idx if winner == defender else battle_idx
            if winner_src != winner_dst:
                _w_inf_arr = self.france_infantry  if winner == FRANCE else self.allies_infantry
                _w_cav_arr = self.france_cavalry   if winner == FRANCE else self.allies_cavalry
                _w_arty_arr = self.france_artillery if winner == FRANCE else self.allies_artillery
                _w_hist_map = self.france_army_history if winner == FRANCE else self.allies_army_history
                snap_w_inf = int(_w_inf_arr[winner_src])
                snap_w_cav = int(_w_cav_arr[winner_src])
                snap_w_arty = int(_w_arty_arr[winner_src])
                snap_w_cmd = winner_cmd_arr[winner_src]
                snap_w_sf = int(self.sub_faction[winner_src]) if winner == ALLIES else SUBFACTION_NONE
                snap_w_hist = list(_w_hist_map.get(winner_src, []))
                # Defer advance (9-tuple) until after _apply_all_moves
                pending_moves.append((winner, winner_src, winner_dst,
                                      snap_w_inf, snap_w_cav, snap_w_arty,
                                      snap_w_cmd, snap_w_sf, snap_w_hist))
            self._vprint(f"  Advance  : {'France' if winner == FRANCE else 'Allies'} → "
                  f"{self.node_ids[winner_dst]}")

            log.append({
                'node':                  nid,
                'attacker':              'France' if attacker == FRANCE else 'Allies',
                'winner':                'France' if winner  == FRANCE else 'Allies',
                'retreat_node':          retreat_node,
                'crossing':              True,
                'france_inf_before':     f_inf,
                'france_cav_before':     f_cav,
                'france_arty_before':    f_arty,
                'allies_inf_before':     a_inf,
                'allies_cav_before':     a_cav,
                'allies_arty_before':    a_arty,
                'france_men_cas':        f_men_cas,
                'france_arty_cas':       f_arty_cas,
                'allies_men_cas':        a_men_cas,
                'allies_arty_cas':       a_arty_cas,
                'france_commander_died': f_cmd_died,
                'allies_commander_died': a_cmd_died,
            })

        return log, cross_f_srcs, cross_a_srcs, pending_moves

    def _resolve_battles(
        self,
        f_moves: List[Tuple[int, int, int]],
        a_moves: List[Tuple[int, int, int]],
    ) -> List[dict]:
        """
        For every node where both factions have men, resolve the battle.
        Attacker = whoever moved there this turn (larger men count wins ties; France default).
        """
        f_moved_to = {dst: src for src, dst, *_ in f_moves if src != dst}
        a_moved_to = {dst: src for src, dst, *_ in a_moves if src != dst}

        log = []
        for i, nid in enumerate(self.node_ids):
            f_men = self._men(FRANCE, i)
            a_men = self._men(ALLIES, i)
            if f_men == 0 or a_men == 0:
                continue

            # Determine attacker. The one who has moved into the node (largest army if both moved into the node in the same turn)
            f_attacked = i in f_moved_to
            a_attacked = i in a_moved_to
            if f_attacked and not a_attacked:
                attacker = FRANCE
            elif a_attacked and not f_attacked:
                attacker = ALLIES
            else:
                attacker = FRANCE if f_men >= a_men else ALLIES

            # Getting the road type for surprise advantage.
            # When both armies moved in simultaneously, if the defender's road is
            # more surprising than the attacker's, set it as primary for no advantage for the attacker
            _road_rank = {'primary': 0, 'secondary': 1, 'tertiary': 2}
            att_src = f_moved_to.get(i) if attacker == FRANCE else a_moved_to.get(i)
            road_type = 'primary'
            if att_src is not None:
                road_type = self._road_type.get((self.node_ids[att_src], nid), 'primary')
            if f_attacked and a_attacked:
                # Both moved in — also check defender's road
                def_src = a_moved_to.get(i) if attacker == FRANCE else f_moved_to.get(i)
                if def_src is not None:
                    def_road = self._road_type.get((self.node_ids[def_src], nid), 'primary')
                    if _road_rank.get(def_road, 0) > _road_rank.get(road_type, 0):
                        road_type = 'primary'

            # Getting subfaction for allies (British, Spanish or Portuguese) for differences in morale and technology advantages
            att_sf = SUBFACTION_NONE if attacker == FRANCE else int(self.sub_faction[i])
            def_sf = int(self.sub_faction[i]) if attacker == FRANCE else SUBFACTION_NONE

            # Getting commanders
            att_cmd = (self.france_commander[i] or 'UNKNOWN') if attacker == FRANCE \
                      else (self.allies_commander[i] or 'UNKNOWN')
            def_cmd = (self.allies_commander[i] or 'UNKNOWN') if attacker == FRANCE \
                      else (self.france_commander[i] or 'UNKNOWN')

            # Getting armies' track record, for momentum advantages
            att_hist = list(self.france_army_history.get(i, []) if attacker == FRANCE
                             else self.allies_army_history.get(i, []))
            def_hist = list(self.allies_army_history.get(i, []) if attacker == FRANCE
                             else self.france_army_history.get(i, []))
            att_wins = sum(att_hist); att_total = len(att_hist)
            def_wins = sum(def_hist); def_total = len(def_hist)

            # Getting factions' track record, for initiative advantages
            f_fact = list(self.france_faction_history)
            a_fact = list(self.allies_faction_history)
            att_fw = sum(f_fact if attacker == FRANCE else a_fact)
            att_ft = len(f_fact if attacker == FRANCE else a_fact)
            def_fw = sum(a_fact if attacker == FRANCE else f_fact)
            def_ft = len(a_fact if attacker == FRANCE else f_fact)

            # Getting supply distances, for logistics advantages
            att_sup = self._supply_distance(i, attacker)
            def_sup = self._supply_distance(i, ALLIES if attacker == FRANCE else FRANCE)


            # Call the battle model
            winner, f_men_cas, a_men_cas, f_arty_cas, a_arty_cas = resolve_battle(
                france_infantry = int(self.france_infantry[i]),
                france_cavalry = int(self.france_cavalry[i]),
                france_artillery = int(self.france_artillery[i]),
                allies_infantry = int(self.allies_infantry[i]),
                allies_cavalry = int(self.allies_cavalry[i]),
                allies_artillery = int(self.allies_artillery[i]),
                terra1 = self._terra1[nid],
                terra2 = self._terra2[nid],
                occ_turns = int(self.occ_turns[i]),
                attacker_faction = attacker,
                attacker_subfaction = att_sf,
                defender_subfaction = def_sf,
                att_commander = att_cmd,
                def_commander = def_cmd,
                road_type = road_type,
                att_supply_dist = att_sup,
                def_supply_dist = def_sup,
                att_army_wins = att_wins,
                att_army_total = att_total,
                def_army_wins = def_wins,
                def_army_total = def_total,
                att_faction_wins = att_fw,
                att_faction_total = att_ft,
                def_faction_wins = def_fw,
                def_faction_total = def_ft,
            )

            # Snapshot pre-battle sizes for the log
            f_inf_snap = int(self.france_infantry[i])
            f_cav_snap = int(self.france_cavalry[i])
            f_arty_snap = int(self.france_artillery[i])
            a_inf_snap = int(self.allies_infantry[i])
            a_cav_snap = int(self.allies_cavalry[i])
            a_arty_snap = int(self.allies_artillery[i])

            # Verbose battle debug print
            _att_q = COMMANDER_QUALITY.get(att_cmd.upper(), COMMANDER_QUALITY_DEFAULT)
            _def_q = COMMANDER_QUALITY.get(def_cmd.upper(), COMMANDER_QUALITY_DEFAULT)
            _sf_name = {-1: 'French', 0: 'British', 1: 'Spanish', 2: 'Portuguese'}
            self._vprint(
                f"\n{'='*62}\n"
                f"  BATTLE at {nid}  (turn {self.turn})\n"
                f"{'='*62}\n"
                f"  Attacker : {'France' if attacker == FRANCE else 'Allies'}"
                f"  [{_sf_name.get(att_sf, '?')}]  cmd={att_cmd} (q={_att_q:.3f})\n"
                f"  Defender : {'France' if attacker != FRANCE else 'Allies'}"
                f"  [{_sf_name.get(def_sf, '?')}]  cmd={def_cmd} (q={_def_q:.3f})\n"
                f"\n── Pre-battle armies ──────────────────────────────────────\n"
                f"  France   inf={f_inf_snap:6d}  cav={f_cav_snap:5d}  arty={f_arty_snap:4d}\n"
                f"  Allies   inf={a_inf_snap:6d}  cav={a_cav_snap:5d}  arty={a_arty_snap:4d}\n"
                f"\n── Model outputs ──────────────────────────────────────────\n"
                f"  WINNER : {'France' if winner == FRANCE else 'Allies'}\n"
                f"  France casualties : men={f_men_cas:5d}  arty={f_arty_cas:3d}\n"
                f"  Allies casualties : men={a_men_cas:5d}  arty={a_arty_cas:3d}\n"
                f"{'='*62}\n"
            )
            # Apply casualties to both sides
            loser = ALLIES if winner == FRANCE else FRANCE
            loser_is_att = (loser == attacker)
            w_men_cas = f_men_cas  if winner == FRANCE else a_men_cas
            w_arty_cas = f_arty_cas if winner == FRANCE else a_arty_cas
            l_men_cas = a_men_cas  if winner == FRANCE else f_men_cas
            l_arty_cas = a_arty_cas if winner == FRANCE else f_arty_cas

            self._apply_casualties(winner, i, w_men_cas, w_arty_cas)
            self._apply_casualties(loser,  i, l_men_cas, l_arty_cas)

            # Winner captures the loser's lost guns
            if winner == FRANCE:
                self.france_artillery[i] += l_arty_cas
            else:
                self.allies_artillery[i] += l_arty_cas

            # Commander post-battle resolution
            f_cmd_died = False
            a_cmd_died = False

            winner_cmd_arr = self.france_commander if winner == FRANCE else self.allies_commander
            loser_cmd_arr = self.allies_commander if winner == FRANCE else self.france_commander
            winner_gone = self.france_gone if winner == FRANCE else self.allies_gone
            loser_gone = self.allies_gone if winner == FRANCE else self.france_gone

            loser_cmd = loser_cmd_arr[i]
            loser_cmd_alive = True
            if loser_cmd and loser_cmd != 'UNKNOWN':
                if self.np_random.random() < COMMANDER_DEATH_PROB:
                    loser_gone.add(loser_cmd)
                    loser_cmd_alive = False
                    replacement = self._pick_commander(loser)
                    loser_cmd_arr[i] = replacement
                    if winner == FRANCE: a_cmd_died = True
                    else:                f_cmd_died = True
                    loser_faction = 'France' if loser == FRANCE else 'Allies'
                    self._vprint(f"  [KIA] {loser_cmd} ({loser_faction}) killed in action"
                          + (f" — replaced by {replacement}" if replacement != 'UNKNOWN'
                             else " — no replacement available"))
                # Surviving loser commander retreats with the army

            winner_cmd = winner_cmd_arr[i]
            if winner_cmd and winner_cmd != 'UNKNOWN':
                if self.np_random.random() < COMMANDER_DEATH_PROB:
                    winner_gone.add(winner_cmd)
                    replacement = self._pick_commander(winner)
                    winner_cmd_arr[i] = replacement
                    if winner == FRANCE: f_cmd_died = True
                    else:                a_cmd_died = True
                    winner_faction = 'France' if winner == FRANCE else 'Allies'
                    self._vprint(f"  [KIA] {winner_cmd} ({winner_faction}) killed in action"
                          + (f" — replaced by {replacement}" if replacement != 'UNKNOWN'
                             else " — no replacement available"))

            # Reset occ_turns when attacker captures the node
            # The winner just took effective control; any defender's prepared
            # fortification bonus must start fresh from the new occupant.
            if winner == attacker:
                self.occ_turns[i] = 0

            # Update battle histories BEFORE retreat
            france_won = (winner == FRANCE)
            self.france_faction_history.append(1 if france_won else 0)
            self.allies_faction_history.append(0 if france_won else 1)
            if winner == FRANCE:
                self.france_army_history.setdefault(i, deque(maxlen=MOMNTA_WINDOW)).append(1)
                self.allies_army_history.setdefault(i, deque(maxlen=MOMNTA_WINDOW)).append(0)
            else:
                self.allies_army_history.setdefault(i, deque(maxlen=MOMNTA_WINDOW)).append(1)
                self.france_army_history.setdefault(i, deque(maxlen=MOMNTA_WINDOW)).append(0)

            # Loser retreat / elimination
            loser_name = 'France' if loser == FRANCE else 'Allies'
            retreat_node = None
            if not self._has_army(loser, i):
                self._vprint(f"  Retreat  : {loser_name} DESTROYED by casualties")
                # Completely destroyed by casualties — release commander if survived
                if loser_cmd_alive and loser_cmd and loser_cmd != 'UNKNOWN':
                    self._release_commander(loser, loser_cmd)
                loser_cmd_arr[i] = None
            elif self._men(loser, i) < MIN_ARMY_SIZE:
                # Remnant too small to remain a fighting force — eliminate it and
                # return its commander (any KIA replacement included) to the pool.
                self._vprint(f"  Retreat  : {loser_name} REMNANT "
                             f"({self._men(loser, i)} < {MIN_ARMY_SIZE} men) — eliminated")
                cur_cmd = loser_cmd_arr[i]
                if cur_cmd and cur_cmd != 'UNKNOWN':
                    self._release_commander(loser, cur_cmd)
                self._wipe_army(loser, i)
                loser_cmd_arr[i] = None
            else:
                retreat_to = self._find_retreat_node(i, loser, loser_is_att, att_src)
                if retreat_to is not None and retreat_to != i:
                    self._retreat_army(loser, i, retreat_to)
                    retreat_node = self.node_ids[retreat_to]
                    self._vprint(f"  Retreat  : {loser_name} → {retreat_node}")
                else:
                    self._vprint(f"  Retreat  : {loser_name} SURROUNDED — disbanded")
                    # Completely surrounded — disband; release commander if survived
                    if loser_cmd_alive and loser_cmd and loser_cmd != 'UNKNOWN':
                        self._release_commander(loser, loser_cmd)
                    self._wipe_army(loser, i)
                    loser_cmd_arr[i] = None

            log.append({
                'node':                  nid,
                'attacker':              'France' if attacker == FRANCE else 'Allies',
                'winner':                'France' if winner  == FRANCE else 'Allies',
                'retreat_node':          retreat_node,
                'france_inf_before':     f_inf_snap,
                'france_cav_before':     f_cav_snap,
                'france_arty_before':    f_arty_snap,
                'allies_inf_before':     a_inf_snap,
                'allies_cav_before':     a_cav_snap,
                'allies_arty_before':    a_arty_snap,
                'france_men_cas':        f_men_cas,
                'france_arty_cas':       f_arty_cas,
                'allies_men_cas':        a_men_cas,
                'allies_arty_cas':       a_arty_cas,
                'france_commander_died': f_cmd_died,
                'allies_commander_died': a_cmd_died,
            })

        return log

    def _resolve_sieges(self) -> List[dict]:
        """
        Garrison attrition each turn.  Men losses split by CAVALRY_CASUALTY_RATIO;
        artillery is not lost to siege attrition.
        """
        log = []
        for i, nid in enumerate(self.node_ids):
            if self._garrison_cap[i] == 0:
                continue
            g = int(self.current_garrison[i])
            if g == 0:
                continue

            f_men = self._men(FRANCE, i)
            a_men = self._men(ALLIES, i)

            def _apply_siege_loss(faction: int, total_men: int):
                a_loss = max(0, round(total_men * SIEGE_ATTACKER_LOSS_RATE))
                cav_loss = round(a_loss * CAVALRY_CASUALTY_RATIO)
                inf_loss = a_loss - cav_loss
                if faction == FRANCE:
                    self.france_cavalry[i]  = max(0, int(self.france_cavalry[i])  - cav_loss)
                    self.france_infantry[i] = max(0, int(self.france_infantry[i]) - inf_loss)
                    if not self._has_army(FRANCE, i):
                        self._release_commander(FRANCE, self.france_commander[i])
                        self.france_commander[i] = None
                else:
                    self.allies_cavalry[i]  = max(0, int(self.allies_cavalry[i])  - cav_loss)
                    self.allies_infantry[i] = max(0, int(self.allies_infantry[i]) - inf_loss)
                    if not self._has_army(ALLIES, i):
                        self._release_commander(ALLIES, self.allies_commander[i])
                        self.allies_commander[i] = None
                # A besieging force ground down below the minimum is eliminated and lifts the siege.
                self._eliminate_remnant(faction, i, cause='siege attrition')
                return a_loss

            if f_men > 0 and a_men == 0 and self.owner[i] != FRANCE:
                g_loss = max(1, int(g * SIEGE_GARRISON_LOSS_RATE))
                a_loss = _apply_siege_loss(FRANCE, g)   # attacker loss scales with garrison, not its own army size
                self.current_garrison[i] = max(0, g - g_loss)

                # If the garrison falls below 100 men, the siege is completed
                if self.current_garrison[i] < 100:
                    self.owner[i] = FRANCE
                    self.current_garrison[i] = int(self._garrison_cap[i])
                    self.occ_turns[i] = 0
                    log.append({'node': nid, 'event': 'captured', 'by': 'France'})
                else:
                    log.append({'node': nid, 'event': 'siege', 'attacker': 'France',
                                'garrison_remaining': int(self.current_garrison[i]),
                                'attacker_loss': a_loss, 'garrison_loss': g_loss})

            elif a_men > 0 and f_men == 0 and self.owner[i] != ALLIES:
                g_loss = max(1, int(g * SIEGE_GARRISON_LOSS_RATE))
                a_loss = _apply_siege_loss(ALLIES, g)   # attacker loss scales with garrison, not army size
                self.current_garrison[i] = max(0, g - g_loss)

                # If the garrison falls below 100 men, the siege is completed
                if self.current_garrison[i] < 100:
                    self.owner[i] = ALLIES
                    self.current_garrison[i] = int(self._garrison_cap[i])
                    self.occ_turns[i] = 0
                    log.append({'node': nid, 'event': 'captured', 'by': 'Allies'})
                else:
                    log.append({'node': nid, 'event': 'siege', 'attacker': 'Allies',
                                'garrison_remaining': int(self.current_garrison[i]),
                                'attacker_loss': a_loss, 'garrison_loss': g_loss})

        return log

    def _update_ownership(self):
        for i in range(self.N):
            f_has = self._has_army(FRANCE, i)
            a_has = self._has_army(ALLIES, i)
            g = int(self.current_garrison[i])

            if f_has and not a_has:
                # France present and unopposed, capture if not already French
                # For garrisoned nodes (cities+) capture only when garrison falls (handled by _resolve_sieges)
                if self.owner[i] != FRANCE and g == 0:
                    self.owner[i] = FRANCE
                    self.occ_turns[i] = 0
            elif a_has and not f_has:
                # Allies present and unopposed
                if self.owner[i] != ALLIES and g == 0:
                    self.owner[i] = ALLIES
                    self.occ_turns[i] = 0
            # When both armies leave, ownership is RETAINED, nodes stay under the last
            # occupying faction until an enemy army physically passes through.

    def _update_occ_turns(self):
        for i in range(self.N):
            if self._has_army(FRANCE, i) != self._has_army(ALLIES, i):
                self.occ_turns[i] += 1


    def _get_reinf_rate(self) -> Tuple[int, int]:
        """
        Return (france_rate, allies_rate_per_depot) for the current turn,
        looked up from REINFORCEMENT_SCHEDULE breakpoints.
        """
        f_rate, a_rate = REINFORCEMENT_SCHEDULE[0][1], REINFORCEMENT_SCHEDULE[0][2]
        for turn_start, fr, ar in REINFORCEMENT_SCHEDULE:
            if self.turn >= turn_start:
                f_rate, a_rate = fr, ar
            else:
                break
        return f_rate, a_rate

    def _step_reinforcements(self):
        """
        Each turn:
          1. Add rate-appropriate men to each depot pool (capped at REINF_POOL_CAP).
          2. If a friendly army is already at the depot -> absorb the ENTIRE pool
             into it immediately (regardless of pool size).
          3. Otherwise, if pool ≥ REINF_SPAWN_THRESHOLD AND faction is under
             MAX_ARMIES cap -> spawn a new army at the depot.
          4. If neither condition is met -> pool keeps growing until cap.

        Depot capture: if the depot node is enemy-owned, no troops accumulate there.

        Reinforcements use _split_army() so they include infantry, cavalry AND
        artillery in the correct historical ratios.
        """
        f_rate, a_rate = self._get_reinf_rate()

        # French depots
        for nid in FRANCE_REINF_DEPOTS:
            if nid not in self.node_idx:
                continue
            i = self.node_idx[nid]

            # No accumulation if depot is enemy-held
            if self.owner[i] == ALLIES:
                continue
            
            # Update pool
            pool = min(
                self.france_reinf_pool.get(nid, 0) + f_rate,
                REINF_POOL_CAP,
            )
            self.france_reinf_pool[nid] = pool

            if pool == 0:
                continue

            if self._has_army(FRANCE, i):
                # Absorb entire pool into the army already present
                inf, cav, arty = self._split_army(pool, FRANCE, SUBFACTION_NONE)
                self.france_infantry[i] += inf
                self.france_cavalry[i] += cav
                self.france_artillery[i] += arty
                self.france_reinf_pool[nid] = 0

            elif pool >= REINF_SPAWN_THRESHOLD and self._count_armies(FRANCE) < MAX_ARMIES:
                # Spawn a fresh army if below the max_armies cap
                inf, cav, arty = self._split_army(pool, FRANCE, SUBFACTION_NONE)
                self.france_infantry[i] = inf
                self.france_cavalry[i] = cav
                self.france_artillery[i] = arty
                self.owner[i] = FRANCE
                self.france_commander[i] = self._pick_commander(FRANCE)
                self.france_army_history[i] = deque(maxlen=MOMNTA_WINDOW)
                self.france_reinf_pool[nid] = 0

        # Allied depots
        for nid in ALLIES_REINF_DEPOTS:
            if nid not in self.node_idx:
                continue
            i = self.node_idx[nid]
            sf = ALLIES_REINF_SUBFACTIONS.get(nid, SUBFACTION_SPANISH)

            # No accumulation if depot is enemy-held
            if self.owner[i] == FRANCE:
                continue

            pool = min(
                self.allies_reinf_pool.get(nid, 0) + a_rate,
                REINF_POOL_CAP,
            )
            self.allies_reinf_pool[nid] = pool

            if pool == 0:
                continue

            if self._has_army(ALLIES, i):
                # Absorb entire pool into the army already present
                inf, cav, arty = self._split_army(pool, ALLIES, sf)
                self.allies_infantry[i] += inf
                self.allies_cavalry[i] += cav
                self.allies_artillery[i] += arty
                # Preserve existing subfaction; only set if currently unassigned
                if self.sub_faction[i] == SUBFACTION_NONE:
                    self.sub_faction[i] = sf
                self.allies_reinf_pool[nid] = 0

            elif pool >= REINF_SPAWN_THRESHOLD and self._count_armies(ALLIES) < MAX_ARMIES:
                # Spawn a fresh army
                inf, cav, arty = self._split_army(pool, ALLIES, sf)
                self.allies_infantry[i]  = inf
                self.allies_cavalry[i] = cav
                self.allies_artillery[i] = arty
                self.owner[i] = ALLIES
                self.sub_faction[i] = sf
                self.allies_commander[i] = self._pick_commander(ALLIES)
                self.allies_army_history[i] = deque(maxlen=MOMNTA_WINDOW)
                self.allies_reinf_pool[nid] = 0



    def _pick_commander(self, faction: int) -> str:
        """
        Remove and return the most senior available commander for this faction.
        Seniority is defined by COMMANDER_SENIORITY (lower number = higher rank).
        Returns 'UNKNOWN' when the pool is empty.
        """
        avail = self.france_available if faction == FRANCE else self.allies_available
        if not avail:
            return 'UNKNOWN'
        best = min(avail,
                   key=lambda c: COMMANDER_SENIORITY.get(c.upper(),
                                                          COMMANDER_SENIORITY_DEFAULT))
        avail.remove(best)
        return best

    def _release_commander(self, faction: int, cmd: Optional[str]):
        """
        Return a commander to the faction's available pool.
        No-op for None, 'UNKNOWN', or commanders in the gone (KIA) set.
        """
        if not cmd or cmd == 'UNKNOWN':
            return
        gone  = self.france_gone  if faction == FRANCE else self.allies_gone
        avail = self.france_available if faction == FRANCE else self.allies_available
        if cmd in gone:
            return   # Dead, never reassignable
        if cmd not in avail:
            avail.append(cmd)


    def _apply_historical_events(self):
        """
        Fire once-per-turn events keyed to self.turn.
        'remove': commander departs — replaced in any army that holds them.
        'add'   : commander becomes available in the pool for future assignments.
        """
        for turn, faction, event, commander in HISTORICAL_COMMANDER_EVENTS:
            if self.turn != turn:
                continue
            if event == 'remove':
                cmd_arr = (self.france_commander if faction == FRANCE
                           else self.allies_commander)
                gone = self.france_gone if faction == FRANCE else self.allies_gone
                gone.add(commander)
                for i in range(self.N):
                    # If the commander to be removed is present in the map, pick a substitute
                    if cmd_arr[i] == commander:
                        replacement = self._pick_commander(faction)
                        cmd_arr[i] = replacement
                        fname = 'France' if faction == FRANCE else 'Allies'
                        self._vprint(f"  [HISTORICAL] {commander} ({fname}) departs at turn {turn}"
                              + (f" — replaced by {replacement}" if replacement != 'UNKNOWN'
                                 else " — no replacement available"))
            elif event == 'add':
                avail = (self.france_available if faction == FRANCE
                         else self.allies_available)
                if commander not in avail:
                    avail.insert(0, commander)   # Front of queue (next to be assigned)
                fname = 'France' if faction == FRANCE else 'Allies'
                self._vprint(f"  [HISTORICAL] {commander} ({fname}) arrives and joins the pool")



    def _territory_consolidation_score(self, faction: int) -> float:
        """
        Average over all owned nodes of:
            strategic_importance[i] * fraction_of_neighbours_also_owned

        Divided by total owned nodes so it stays comparable regardless of
        how much territory the faction holds.  Returns 0 if no nodes owned.
        """
        owned = [i for i in range(self.N) if self.owner[i] == faction]
        if not owned:
            return 0.0
        total = 0.0
        for i in owned:
            nid = self.node_ids[i]
            nbrs = [n for n in self.neighbour_list[nid] if n is not None]
            frac = (sum(1 for n in nbrs if self.owner[self.node_idx[n]] == faction) / len(nbrs)
                    if nbrs else 1.0)
            total += float(self._strategic_importance[i]) * frac
        return total / len(owned)

    def _army_consolidation_score(self, faction: int) -> float:
        """
        Weighted average of neighbour_ownership_fraction across all nodes
        that hold a friendly army, weighted by each army's share of total
        faction troops.  Already in [0, 1] by construction (weights sum to 1).
        Returns 0 if the faction has no troops.
        """
        if faction == FRANCE:
            inf_arr, cav_arr = self.france_infantry, self.france_cavalry
        else:
            inf_arr, cav_arr = self.allies_infantry, self.allies_cavalry

        total_men = max(1, int(np.sum(inf_arr + cav_arr)))
        score = 0.0
        for i in range(self.N):
            men = int(inf_arr[i] + cav_arr[i])
            if men == 0:
                continue
            nid = self.node_ids[i]
            nbrs = [n for n in self.neighbour_list[nid] if n is not None]
            frac = (sum(1 for n in nbrs if self.owner[self.node_idx[n]] == faction) / len(nbrs)
                    if nbrs else 1.0)
            score += (men / total_men) * frac
        return score

    def _check_dominance(self) -> int:
        """
        Returns FRANCE or ALLIES if one side has achieved strategic dominance,
        otherwise returns NEUTRAL (neither side has won yet).

        Dominance requires simultaneously:
          - All nodes with SI >= 4 owned by that faction
          - >= 90% of nodes with SI == 3 (cities) owned by that faction
          - >= 75% of nodes with SI == 2 (towns) owned by that faction
          - the OPPOSING faction reduced below DOMINANCE_TROOP_FLOOR troops
            across the entire map (otherwise the Allies' 1808 territorial
            position alone would end the game on turn 1)
        """
        france_troops = int(np.sum(self.france_infantry + self.france_cavalry + self.france_artillery))
        allies_troops = int(np.sum(self.allies_infantry + self.allies_cavalry + self.allies_artillery))

        for faction in (FRANCE, ALLIES):
            mask_high = self._si_raw >= 4
            mask_three = self._si_raw == 3
            mask_two = self._si_raw == 2

            owns_high = int(np.sum((self.owner == faction) & mask_high))
            owns_three = int(np.sum((self.owner == faction) & mask_three))
            owns_two = int(np.sum((self.owner == faction) & mask_two))

            if (self._n_si_high > 0 and owns_high < self._n_si_high) :
                continue
            if (self._n_si_three > 0 and owns_three < 0.90 * self._n_si_three):
                continue
            if (self._n_si_two > 0 and owns_two < 0.75 * self._n_si_two):
                continue

            # Territorial condition met — only a win if the loser is also militarily spent
            loser_troops = allies_troops if faction == FRANCE else france_troops
            if loser_troops >= DOMINANCE_TROOP_FLOOR:
                continue

            return faction
        return NEUTRAL

    def _terminal_score(self) -> float:
        """
        France score minus Allies score at game end.
        Score = sum over owned nodes of terminal_weight[node_type].

        Weights:  capital=4, regional_capital=2, major_city=1,
                  city=0.2,  town=0.05, intersection=0
        """
        france_score = float(np.sum(self._terminal_weight[self.owner == FRANCE]))
        allies_score = float(np.sum(self._terminal_weight[self.owner == ALLIES]))
        return france_score - allies_score

    def _compute_reward(
        self,
        prev_owner: np.ndarray,
        battle_log: List[dict],
        siege_log:  List[dict],
    ) -> float:
        """
        Compute France's scalar reward for this step.
        Allies reward = -result  (zero-sum by construction).

        Terminal win/loss is added separately in step() after this returns.
        """
        reward = 0.0

        # 1. Node delta — weighted by per-type capture value (config.NODE_CAPTURE_VALUE)
        for i in range(self.N):
            if self.owner[i] == prev_owner[i]:
                # No captured this turn
                continue
            cv = float(self._capture_value[i])
            if self.owner[i] == FRANCE:
                reward += cv * W_NODE_DELTA
            elif prev_owner[i] == FRANCE:
                reward -= cv * W_NODE_DELTA

        # 2. Casualty delta — net men (enemy losses − own losses) from battles
        france_cas = sum(b['france_men_cas'] for b in battle_log)
        allies_cas = sum(b['allies_men_cas'] for b in battle_log)
        # Siege losses: attacker pays attacker_loss, defender garrison pays garrison_loss.
        for s in siege_log:
            if s.get('event') == 'siege':
                g_loss = s.get('garrison_loss', 0)
                if s['attacker'] == 'France':
                    france_cas += s['attacker_loss']
                    allies_cas += g_loss            
                else:
                    allies_cas += s['attacker_loss']
                    france_cas += g_loss
        reward += (allies_cas - france_cas) * W_CASUALTY

        # 3. Battle outcome per battle, winner earns strategic importance of node
        for b in battle_log:
            si = float(self._strategic_importance[self.node_idx[b['node']]])
            reward += si * W_BATTLE_OUTCOME if b['winner'] == 'France' else -si * W_BATTLE_OUTCOME

        # 4. Territory consolidation (stock)
        f_tc = self._territory_consolidation_score(FRANCE)
        a_tc = self._territory_consolidation_score(ALLIES)
        reward += (f_tc - a_tc) * W_TERRITORY_CONSOL

        # 5. Army consolidation (stock)
        f_ac = self._army_consolidation_score(FRANCE)
        a_ac = self._army_consolidation_score(ALLIES)
        reward += (f_ac - a_ac) * W_ARMY_CONSOL

        # 6. Enemy depots held (stock)
        for nid in ALLIES_REINF_DEPOTS:
            if nid in self.node_idx and self.owner[self.node_idx[nid]] == FRANCE:
                reward += W_ENEMY_DEPOT
        for nid in FRANCE_REINF_DEPOTS:
            if nid in self.node_idx and self.owner[self.node_idx[nid]] == ALLIES:
                reward -= W_ENEMY_DEPOT

        return reward

    def _depot_penalty(self, faction: int) -> float:
        """
        Independent (non-zero-sum) self-penalty for letting own reinforcement
        depot pools exceed the spawn threshold.

        Gradient grows from 20k at 40k to W_DEPOT_BASE_PENALTY at 40k, plus a
        discrete W_DEPOT_SPIKE once the pool hits the 40k cap (at which point
        troops are actively discarded each turn).

        Applied separately to each faction in step() — NOT part of the
        zero-sum reward, because an agent cannot influence the opponent's
        depot management and should not receive noise for it.
        """
        depots = FRANCE_REINF_DEPOTS if faction == FRANCE else ALLIES_REINF_DEPOTS
        pool_map = self.france_reinf_pool if faction == FRANCE else self.allies_reinf_pool
        threshold = float(REINF_SPAWN_THRESHOLD)
        cap = float(REINF_POOL_CAP)
        penalty = 0.0
        for nid in depots:
            pool = float(pool_map.get(nid, 0))
            if pool > threshold:
                gradient = (pool - threshold) / threshold   # 0 → 1
                penalty += gradient * W_DEPOT_BASE_PENALTY
                if pool >= cap:
                    penalty += W_DEPOT_SPIKE
        return penalty


    
    # Observation builder

    def _get_obs(self, faction: int) -> np.ndarray:
        """
        This is for TIER 1 architecture. For Tier 2, _build_local_obs from sb3_wrapper.py is the observation constructor.
        Build the float32 observation vector for one faction.

        Layout: N * 43 per-node features  +  11 global features.
        Total shape: 231 * 43 + 11 = 9,944  (for the default 231-node map).
        Nodes outside the faction's fog-of-war radius are zeroed out entirely.

        Per-node (43 values, offset = i * 43):
          [0]     owner_norm          0=neutral, 0.5=France, 1.0=Allies
          [1]     france_infantry_norm
          [2]     france_cavalry_norm
          [3]     france_artillery_norm
          [4]     allies_infantry_norm
          [5]     allies_cavalry_norm
          [6]     allies_artillery_norm
          [7]     occ_turns_norm       clamped [0,1] over 52 turns
          [8]     garrison_norm        current_garrison / MAX_GARRISON_OBS
          [9]     allies_subfaction    0=none 0.33=BR 0.67=SP 1.0=PT
          [11]    allies_commander_q   quality [0,1], 0 if absent
          [12]    visible              1.0 if inside fog radius
          [13]    france_army_wins_norm    sum(france_army_history[i]) / MOMNTA_WINDOW
          [14]    france_army_total_norm   len(france_army_history[i]) / MOMNTA_WINDOW
          [15]    allies_army_wins_norm
          [16]    allies_army_total_norm
          [17]    depot_pool_norm      pool at this node / REINF_POOL_CAP (0 for non-depots)
          [18]    strategic_importance_norm  (1-6)/6
          [19-42] 8 neighbours x 3 values each:
                    nbr_idx / (N-1)   (0 if no neighbour)
                    dist_norm         distance_km / 300  (0 if no neighbour)
                    road_type_enc     primary=1.0, secondary=0.5, tertiary=0.25, none=0.0

        Global (11 values, offset = N * 43):
          [g+0]  turn_norm
          [g+1]  france_share         france_men / total_men
          [g+2]  allies_share         allied_men / total_men
          [g+3]  france_pool_norm     aggregate depot pool / (REINF_POOL_CAP * n_depots)
          [g+4]  allies_pool_norm
          [g+5]  france_inita_wins_norm   sum(france_faction_history) / INITA_WINDOW
          [g+6]  france_inita_total_norm  len(france_faction_history) / INITA_WINDOW
          [g+7]  allies_inita_wins_norm
          [g+8]  allies_inita_total_norm
          [g+9]  france_army_count_norm  active france armies / MAX_ARMIES
          [g+10] allies_army_count_norm  active allies armies / MAX_ARMIES
        """
        OBS_LEN = self.N * 43 + 11
        obs = np.zeros(OBS_LEN, dtype=np.float32)

        # Fog-of-war: mark visible nodes
        inf_own = self.france_infantry if faction == FRANCE else self.allies_infantry
        cav_own = self.france_cavalry  if faction == FRANCE else self.allies_cavalry

        seed_lats: list = []
        seed_lons: list = []
        for i, nid in enumerate(self.node_ids):
            if self.owner[i] == faction or int(inf_own[i]) + int(cav_own[i]) > 0:
                seed_lats.append(self._lat[nid])
                seed_lons.append(self._lon[nid])

        visible = np.zeros(self.N, dtype=bool)
        for i, nid in enumerate(self.node_ids):
            lat_i = self._lat[nid]
            lon_i = self._lon[nid]
            for slat, slon in zip(seed_lats, seed_lons):
                if _haversine(slat, slon, lat_i, lon_i) <= FOG_RADIUS_KM:
                    visible[i] = True
                    break

        # Normalisation constants
        _INF_NORM = 100_000.0
        _CAV_NORM = 15_000.0   # Max realistic cavalry at one node (15% of 100k)
        _ARTY_NORM = 300.0    # 3 guns/1k men × 100k cap = 300 guns
        _OCC_NORM = 52.0
        _DIST_NORM = 300.0
        _ROAD_ENC = {'primary': 1.0, 'secondary': 0.5, 'tertiary': 0.25}
        _N1 = float(max(1, self.N - 1))

        f_men_total = float(max(1, int(np.sum(self.france_infantry + self.france_cavalry))))
        a_men_total = float(max(1, int(np.sum(self.allies_infantry + self.allies_cavalry))))
        grand_total = max(1.0, f_men_total + a_men_total)

        # Faction-level depot pool maps for per-node depot_pool_norm
        france_depot_pool: Dict[int, float] = {
            self.node_idx[nid]: self.france_reinf_pool.get(nid, 0) / float(REINF_POOL_CAP)
            for nid in FRANCE_REINF_DEPOTS if nid in self.node_idx
        }
        allies_depot_pool: Dict[int, float] = {
            self.node_idx[nid]: self.allies_reinf_pool.get(nid, 0) / float(REINF_POOL_CAP)
            for nid in ALLIES_REINF_DEPOTS if nid in self.node_idx
        }

        # Per-node features
        for i, nid in enumerate(self.node_ids):
            base = i * 43
            owner = int(self.owner[i])

            # [0] owner
            obs[base + 0] = 0.0 if owner == NEUTRAL else 0.5 if owner == FRANCE else 1.0

            # [18] node's strategic importance
            obs[base + 18] = float(self._strategic_importance[i])

            # [19-42] neighbour edge features: 8 slots x 3 values
            nbrs = self.neighbour_list[nid]
            dists = self.neighbour_dist[nid]
            for d in range(MAX_DEGREE):
                nbr_name = nbrs[d] if d < len(nbrs) else None
                slot = base + 19 + d * 3
                if nbr_name is None:
                    continue
                nbr_idx = self.node_idx[nbr_name]
                dist_km = dists[d] if (d < len(dists) and dists[d] is not None) else 0.0
                rt = self._road_type.get((nid, nbr_name), '')
                obs[slot] = nbr_idx / _N1
                obs[slot + 1] = min(1.0, dist_km / _DIST_NORM)
                obs[slot + 2] = _ROAD_ENC.get(rt, 0.0)


            if not visible[i]:
                continue   # Only knows the owner, strategic importance and neighbours of each node,
                                                    # not the information within each (fog of war)

            # [1-6] troop counts
            obs[base + 1] = min(1.0, int(self.france_infantry[i])  / _INF_NORM)
            obs[base + 2] = min(1.0, int(self.france_cavalry[i])   / _CAV_NORM)
            obs[base + 3] = min(1.0, int(self.france_artillery[i]) / _ARTY_NORM)
            obs[base + 4] = min(1.0, int(self.allies_infantry[i])  / _INF_NORM)
            obs[base + 5] = min(1.0, int(self.allies_cavalry[i])   / _CAV_NORM)
            obs[base + 6] = min(1.0, int(self.allies_artillery[i]) / _ARTY_NORM)

            # [7] occupation turns
            obs[base + 7] = min(1.0, int(self.occ_turns[i]) / _OCC_NORM)

            # [8] garrison strength
            obs[base + 8] = min(1.0, int(self.current_garrison[i])
                                  / float(max(1, MAX_GARRISON_OBS)))

            # [9] allied subfaction
            obs[base + 9] = _subfaction_norm(int(self.sub_faction[i]))

            # [10-11] commander quality
            f_cmd = self.france_commander[i]
            a_cmd = self.allies_commander[i]
            obs[base + 10] = (
                COMMANDER_QUALITY.get(f_cmd.upper(), COMMANDER_QUALITY_DEFAULT)
                if f_cmd and f_cmd != 'UNKNOWN' else 0.0
            )
            obs[base + 11] = (
                COMMANDER_QUALITY.get(a_cmd.upper(), COMMANDER_QUALITY_DEFAULT)
                if a_cmd and a_cmd != 'UNKNOWN' else 0.0
            )

            # [12] visible flag
            obs[base + 12] = 1.0

            # [13-16] army track records (MOMNTA_WINDOW)
            f_hist = self.france_army_history.get(i)
            a_hist = self.allies_army_history.get(i)
            if f_hist:
                obs[base + 13] = sum(f_hist) / float(MOMNTA_WINDOW)
                obs[base + 14] = len(f_hist) / float(MOMNTA_WINDOW)
            if a_hist:
                obs[base + 15] = sum(a_hist) / float(MOMNTA_WINDOW)
                obs[base + 16] = len(a_hist) / float(MOMNTA_WINDOW)

            # [17] per-node depot pool (own faction's pool at this node)
            if faction == FRANCE:
                obs[base + 17] = min(1.0, france_depot_pool.get(i, 0.0))
            else:
                obs[base + 17] = min(1.0, allies_depot_pool.get(i, 0.0))


        # Global features
        g = self.N * 43

        # [9890] Turn
        obs[g + 0] = min(1.0, self.turn / float(max(1, MAX_TURNS)))

        # [9891-92] Faction troops ratio
        obs[g + 1] = f_men_total / grand_total
        obs[g + 2] = a_men_total / grand_total

        # [9893-94] Faction depot pool ratio
        f_pool_max = float(REINF_POOL_CAP * max(1, len(FRANCE_REINF_DEPOTS)))
        a_pool_max = float(REINF_POOL_CAP * max(1, len(ALLIES_REINF_DEPOTS)))
        obs[g + 3] = min(1.0, sum(self.france_reinf_pool.values()) / f_pool_max)
        obs[g + 4] = min(1.0, sum(self.allies_reinf_pool.values()) / a_pool_max)


        # [9895-98] Faction's battle history over last 5 battles
        f_inita = self.france_faction_history
        a_inita = self.allies_faction_history
        obs[g + 5] = sum(f_inita) / float(INITA_WINDOW) if f_inita else 0.0
        obs[g + 6] = len(f_inita) / float(INITA_WINDOW)
        obs[g + 7] = sum(a_inita) / float(INITA_WINDOW) if a_inita else 0.0
        obs[g + 8] = len(a_inita) / float(INITA_WINDOW)

        # [9899-9900] Faction's armies ratio
        obs[g + 9]  = self._count_armies(FRANCE) / float(MAX_ARMIES)
        obs[g + 10] = self._count_armies(ALLIES)  / float(MAX_ARMIES)

        return obs


    def _get_info(self) -> dict:
        f_men = int(np.sum(self.france_infantry + self.france_cavalry))
        a_men = int(np.sum(self.allies_infantry  + self.allies_cavalry))
        f_arty = int(np.sum(self.france_artillery))
        a_arty = int(np.sum(self.allies_artillery))
        info = {
            'turn': self.turn,
            'france_troops': f_men,
            'france_arty': f_arty,
            'allies_troops': a_men,
            'allies_arty': a_arty,
            'france_nodes': int(np.sum(self.owner == FRANCE)),
            'allies_nodes': int(np.sum(self.owner == ALLIES)),
            'neutral_nodes': int(np.sum(self.owner == NEUTRAL)),
            'france_reinf_pools': dict(self.france_reinf_pool),
            'allies_reinf_pools': dict(self.allies_reinf_pool),
        }

        # Stats for end-of-episode W&B logging
        # Total troops including artillery, active army counts, and node control
        # broken down by node type — for BOTH factions.
        info['france_troops_total'] = f_men + f_arty
        info['allies_troops_total'] = a_men + a_arty
        info['france_armies'] = self._count_armies(FRANCE)
        info['allies_armies'] = self._count_armies(ALLIES)
        france_owned = (self.owner == FRANCE)
        allies_owned = (self.owner == ALLIES)
        for t in self._NODE_TYPES:
            mask = self._type_masks[t]
            info[f'france_nodes_{t}'] = int(np.sum(france_owned & mask))
            info[f'allies_nodes_{t}'] = int(np.sum(allies_owned & mask))
        return info

    def _post_label(self, node_idx: int) -> str:
        """Return the positional-defence label for a node: HD / PD / FD."""
        t = self.occ_turns[node_idx]
        if t < 2:  return 'HD'
        if t < 8:  return 'PD'
        return 'FD'
