from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from .config import (
    FRANCE, ALLIES,
    TECHA_TABLE, MORALA_TABLE,
    COMMANDER_QUALITY, COMMANDER_QUALITY_DEFAULT,
    SUBFACTION_NONE,
    LOGSA_THRESHOLDS,
    LOPSIDED_RATIO,
    LOPSIDED_WINNER_CAS_RATE,
    LOPSIDED_LOSER_CAS_RATE
)


# Load models once at import time

_root = Path(__file__).parent.parent / 'Battle_Outcome_Model'

try:
    import joblib
    _wina_bundle       = joblib.load(_root / 'battle_outcome_wina.pkl')
    _casualties_bundle = joblib.load(_root / 'battle_outcome_casualties.pkl')
    _MODELS_LOADED = True
except Exception as e:
    print(f'[battle_model] WARNING: could not load ML models ({e}). '
          f'Falling back to stub resolver.')
    _MODELS_LOADED = False



def _occ_turns_to_post1(occ_turns: int) -> str:
    """
    Derives post1.

    0-2 turns (0-2 weeks): HD — Hasty Defence  (army just arrived or barely settled)
    3-7 turns (3-7 weeks): PD — Prepared Defence
    8+ turns  (8+ weeks):  FD — Fortified Defence
    """
    if occ_turns < 2: return 'HD'
    if occ_turns < 8: return 'PD'
    return 'FD'


def _derive_techa(attacker_faction: int, defender_subfaction: int) -> int:
    """
    Derives techa.
     
    Based on the attacker and defender's factions.
    """
    return TECHA_TABLE.get((attacker_faction, defender_subfaction), 0)


def _derive_morala(
    attacker_subfaction: int,
    defender_subfaction: int,
) -> int:
    """
    Derives morala.

    Spanish and Portuguese have a home-soil base of 1; British and French 0.
    """
    return MORALA_TABLE.get((attacker_subfaction, defender_subfaction), 0)


def _derive_logsa(att_supply_dist: float, def_supply_dist: float) -> int:
    """
    If the difference in distance to the nearest supply depot between the defender and the attacker is less than 100 km, logsa
    will be 0. If between 100 and 250 km, it will be 1. More than that, it will be 2.
    """
    diff = def_supply_dist - att_supply_dist
    lo, hi = LOGSA_THRESHOLDS
    if diff >= hi: return 2
    if diff >= lo: return 1
    return 0


def _derive_momnta(att_wins: int, att_total: int, def_wins: int, def_total: int) -> int:
    """
    Derive momnta.

    A difference on the number of wins of the last three engagements of each army.
    """
    att_w = att_wins if att_total > 0 else 0
    def_w = def_wins if def_total > 0 else 0
    return 1 if (att_w - def_w) >= 2 else 0


def _derive_inita(
    att_faction_wins: int, att_faction_total: int,
    def_faction_wins: int, def_faction_total: int,
) -> int:
    """
    Derive inita.
    
    A difference on the number of wins of the last five engagements of each faction.
    """
    aw = att_faction_wins if att_faction_total > 0 else 0
    dw = def_faction_wins if def_faction_total > 0 else 0
    return max(0, min(2, (aw - dw) // 2))


def _derive_surpa(road_type: str) -> int:
    """
    Derive surpa.

    Depends on roadtype.
    """
    return {'primary': 0, 'secondary': 1, 'tertiary': 2}.get(road_type, 0)


def _build_feature_row(
    attacker_faction:     int,
    # ── Actual unit counts (not derived from ratios) ──────────────────────
    france_infantry:      int,
    france_cavalry:       int,
    france_artillery:     int,
    allies_infantry:      int,
    allies_cavalry:       int,
    allies_artillery:     int,
    # ── Terrain / posture ────────────────────────────────────────────────
    terra1:               str,
    terra2:               str,
    occ_turns:            int,
    # ── Context features ─────────────────────────────────────────────────
    attacker_subfaction:  int   = SUBFACTION_NONE,
    defender_subfaction:  int   = SUBFACTION_NONE,
    att_commander:        str   = 'UNKNOWN',
    def_commander:        str   = 'UNKNOWN',
    road_type:            str   = 'primary',
    att_supply_dist:      float = 0.0,
    def_supply_dist:      float = 0.0,
    att_army_wins:        int   = 0,
    att_army_total:       int   = 0,
    def_army_wins:        int   = 0,
    def_army_total:       int   = 0,
    att_faction_wins:     int   = 0,
    att_faction_total:    int   = 0,
    def_faction_wins:     int   = 0,
    def_faction_total:    int   = 0,
) -> dict:
    """
    Build the feature dict expected by both ML pipelines.
    attacker_str / defender_str = infantry + cavalry (men only).
    attacker_cav / attacker_arty are the actual unit counts, not ratio-derived.
    """
    is_france_att = (attacker_faction == FRANCE)

    att_inf  = france_infantry  if is_france_att else allies_infantry
    att_cav  = france_cavalry   if is_france_att else allies_cavalry
    att_arty = france_artillery if is_france_att else allies_artillery
    def_inf  = allies_infantry  if is_france_att else france_infantry
    def_cav  = allies_cavalry   if is_france_att else france_cavalry
    def_arty = allies_artillery if is_france_att else france_artillery

    att_size = att_inf + att_cav
    def_size = def_inf + def_cav

    return dict(
        attacker_str   = att_size,
        attacker_cav   = att_cav,
        attacker_arty  = att_arty,
        defender_str   = def_size,
        defender_cav   = def_cav,
        defender_arty  = def_arty,
        post1          = _occ_turns_to_post1(occ_turns),
        terra1         = terra1,
        terra2         = terra2,
        surpa          = _derive_surpa(road_type),
        morala         = _derive_morala(attacker_subfaction, defender_subfaction),
        logsa          = _derive_logsa(att_supply_dist, def_supply_dist),
        momnta         = _derive_momnta(att_army_wins, att_army_total,
                                        def_army_wins, def_army_total),
        techa          = _derive_techa(attacker_faction, defender_subfaction),
        inita          = _derive_inita(att_faction_wins, att_faction_total,
                                       def_faction_wins, def_faction_total),
        attacker_pri1    = 'FF',   # Hardcoded as Frontal Assault, as a vast majority of the Battle Dataset had this value
        defender_pri1    = 'DD',   # Harcoded as Defensive Plan
        att_comm_quality = COMMANDER_QUALITY.get(att_commander.upper(), COMMANDER_QUALITY_DEFAULT),
        def_comm_quality = COMMANDER_QUALITY.get(def_commander.upper(), COMMANDER_QUALITY_DEFAULT),
    )



def _ml_resolve(
    attacker_faction:  int,
    france_infantry:   int,
    france_cavalry:    int,
    france_artillery:  int,
    allies_infantry:   int,
    allies_cavalry:    int,
    allies_artillery:  int,
    row:               dict,
) -> tuple[int, int, int, int, int]:
    """
    Run all four ML sub-models.
    Returns (winner, france_men_cas, allies_men_cas, france_arty_cas, allies_arty_cas).
    """
    df_row = pd.DataFrame([row])
    is_france_att = (attacker_faction == FRANCE)

    # Predict Winner
    proba = _wina_bundle['model'].predict_proba(
                _wina_bundle['pipeline'].transform(df_row))[0]
    wina  = int(proba[1] > _wina_bundle['threshold'])
    winner = attacker_faction if wina else (ALLIES if attacker_faction == FRANCE else FRANCE)

    # Men casualties (fraction of men strength)
    att_log  = _casualties_bundle['att_cas']['model'].predict(
                   _casualties_bundle['att_cas']['pipeline'].transform(df_row))[0]
    def_frac = _casualties_bundle['def_cas']['model'].predict(
                   _casualties_bundle['def_cas']['pipeline'].transform(df_row))[0]

    att_men_frac = float(np.expm1(att_log))   # invert log1p, as it was used in the attacker men's casualties model
    def_men_frac = float(def_frac)

    att_men = (france_infantry + france_cavalry) if is_france_att else (allies_infantry + allies_cavalry)
    def_men = (allies_infantry + allies_cavalry) if is_france_att else (france_infantry + france_cavalry)

    att_men_cas = max(0, round(max(0, att_men_frac) * att_men))
    def_men_cas = max(0, round(max(0, def_men_frac) * def_men))

    # Artillery casualties (guns lost, direct count)
    att_arty_cas = max(0, round(float(
        _casualties_bundle['att_carty']['model'].predict(
            _casualties_bundle['att_carty']['pipeline'].transform(df_row))[0]
    )))
    def_arty_cas = max(0, round(float(
        _casualties_bundle['def_carty']['model'].predict(
            _casualties_bundle['def_carty']['pipeline'].transform(df_row))[0]
    )))

    # Clamp arty casualties to actual guns present
    att_arty = france_artillery if is_france_att else allies_artillery
    def_arty = allies_artillery if is_france_att else france_artillery
    att_arty_cas = min(att_arty_cas, att_arty)
    def_arty_cas = min(def_arty_cas, def_arty)

    # Map attacker/defender -> france/allies
    f_men_cas  = att_men_cas  if is_france_att else def_men_cas
    a_men_cas  = def_men_cas  if is_france_att else att_men_cas
    f_arty_cas = att_arty_cas if is_france_att else def_arty_cas
    a_arty_cas = def_arty_cas if is_france_att else att_arty_cas

    return winner, f_men_cas, a_men_cas, f_arty_cas, a_arty_cas




def _lopsided_resolve(
    attacker_faction:  int,
    france_infantry:   int,
    france_cavalry:    int,
    france_artillery:  int,
    allies_infantry:   int,
    allies_cavalry:    int,
    allies_artillery:  int,
) -> tuple[int, int, int, int, int] | None:
    """
    The ML model was trained on historical Peninsular War battles where extreme
    strength imbalances never occurred (no rational commander fought 10:1 odds —
    they retreated).  When the simulation produces such battles (e.g. a
    detachment stumbling into a full army), the model extrapolates outside its
    training distribution and gives nonsensical results.

    Because of this, we introduced a fix. If the strength ratio exceeds LOPSIDED_RATIO, bypass the ML model and
    resolve deterministically — the larger side wins, with casualties scaled to
    reflect a brief, one-sided engagement.

    Casualty logic for the lopsided case:
        Winner  → light losses  (LOPSIDED_WINNER_CAS_RATE  × their men)
        Loser   → heavy losses  (LOPSIDED_LOSER_CAS_RATE   × their men)
        Arty    → same fractional rate as men for each side
    """

    f_men = france_infantry + france_cavalry
    a_men = allies_infantry + allies_cavalry

    larger  = max(f_men, a_men)
    smaller = min(f_men, a_men)
    if smaller == 0 or (larger / smaller) < LOPSIDED_RATIO:
        return None   # Within normal range — ML model decides

    # Larger side wins
    winner     = FRANCE if f_men >= a_men else ALLIES
    loser_men  = a_men if winner == FRANCE else f_men
    winner_men = f_men if winner == FRANCE else a_men

    # Both sides' casualties scale with the loser's strength: the winner can
    # only take as many hits as the (smaller) enemy force is capable of dealing.
    winner_cas = round(loser_men * LOPSIDED_WINNER_CAS_RATE)
    loser_cas  = round(loser_men * LOPSIDED_LOSER_CAS_RATE)

    # Artillery: derive a fractional rate from the men casualties for each side
    w_arty_rate = winner_cas / max(1, winner_men)
    l_arty_rate = loser_cas  / max(1, loser_men)

    if winner == FRANCE:
        f_men_cas  = winner_cas;  a_men_cas  = loser_cas
        f_arty_cas = min(france_artillery, round(france_artillery * w_arty_rate))
        a_arty_cas = min(allies_artillery, round(allies_artillery * l_arty_rate))
    else:
        f_men_cas  = loser_cas;   a_men_cas  = winner_cas
        f_arty_cas = min(france_artillery, round(france_artillery * l_arty_rate))
        a_arty_cas = min(allies_artillery, round(allies_artillery * w_arty_rate))

    return winner, f_men_cas, a_men_cas, f_arty_cas, a_arty_cas



def resolve_battle(
    france_infantry:      int,
    france_cavalry:       int,
    france_artillery:     int,
    allies_infantry:      int,
    allies_cavalry:       int,
    allies_artillery:     int,
    terra1:               str   = 'R',
    terra2:               str   = 'M',
    occ_turns:            int   = 0,
    attacker_faction:     int   = FRANCE,
    attacker_subfaction:  int   = SUBFACTION_NONE,
    defender_subfaction:  int   = SUBFACTION_NONE,
    att_commander:        str   = 'UNKNOWN',
    def_commander:        str   = 'UNKNOWN',
    road_type:            str   = 'primary',
    att_supply_dist:      float = 0.0,
    def_supply_dist:      float = 0.0,
    att_army_wins:        int   = 0,
    att_army_total:       int   = 0,
    def_army_wins:        int   = 0,
    def_army_total:       int   = 0,
    att_faction_wins:     int   = 0,
    att_faction_total:    int   = 0,
    def_faction_wins:     int   = 0,
    def_faction_total:    int   = 0,
    **kwargs,
) -> tuple[int, int, int, int, int]:
    """
    Resolve a battle between France and Allies at a single node.

    Returns:
        winner           : FRANCE (0) or ALLIES (1)
        france_men_cas   : French infantry+cavalry losses (men)
        allies_men_cas   : Allied infantry+cavalry losses (men)
        france_arty_cas  : French artillery losses (guns)
        allies_arty_cas  : Allied artillery losses (guns)

    Peninsular_war_env) is responsible for splitting men casualties
    into infantry vs cavalry using CAVALRY_CASUALTY_RATIO.
    """
    row = _build_feature_row(
        attacker_faction    = attacker_faction,
        france_infantry     = france_infantry,
        france_cavalry      = france_cavalry,
        france_artillery    = france_artillery,
        allies_infantry     = allies_infantry,
        allies_cavalry      = allies_cavalry,
        allies_artillery    = allies_artillery,
        terra1              = terra1,
        terra2              = terra2,
        occ_turns           = occ_turns,
        attacker_subfaction = attacker_subfaction,
        defender_subfaction = defender_subfaction,
        att_commander       = att_commander,
        def_commander       = def_commander,
        road_type           = road_type,
        att_supply_dist     = att_supply_dist,
        def_supply_dist     = def_supply_dist,
        att_army_wins       = att_army_wins,
        att_army_total      = att_army_total,
        def_army_wins       = def_army_wins,
        def_army_total      = def_army_total,
        att_faction_wins    = att_faction_wins,
        att_faction_total   = att_faction_total,
        def_faction_wins    = def_faction_wins,
        def_faction_total   = def_faction_total,
    )

    # Lopsided-battle check (before ML model)
    # Bypasses the ML model when the strength ratio is extreme
    override = _lopsided_resolve(
        attacker_faction,
        france_infantry, france_cavalry, france_artillery,
        allies_infantry, allies_cavalry, allies_artillery,
    )
    if override is not None:
        return override

    args = (attacker_faction,
            france_infantry, france_cavalry, france_artillery,
            allies_infantry, allies_cavalry, allies_artillery,
            row)

    return _ml_resolve(*args)
