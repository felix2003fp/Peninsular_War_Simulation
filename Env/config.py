# Faction IDs 
FRANCE = 0
ALLIES = 1
NEUTRAL = -1

# Sub-faction IDs
SUBFACTION_NONE = -1   # French armies
SUBFACTION_BRITISH = 0
SUBFACTION_SPANISH = 1
SUBFACTION_PORTUGUESE = 2

# Turn / time
MAX_TURNS = 313   # ~6 years at 1 turn/week (1808-1814)
TURNS_PER_YEAR = 52

# Victory / dominance
# A territorial dominance win (see _check_dominance) only counts if the LOSING
# side has been ground down below this many troops across the whole map.
# Without this gate the Allies already satisfy the territorial condition at the
# 1808 start (they hold every important node), so games ends on turn 1.
DOMINANCE_TROOP_FLOOR = 25_000

# Fog of war
FOG_RADIUS_KM = 120.0   # Visibility radius from any owned node or army

# Action space 
MAX_DEGREE = 8     # Max neighbours per node (highest degree in map)
MAX_ARMIES = 10    # Hard cap on concurrent armies per faction

# Minimum viable army size. After a battle, a loser left with fewer than this
# many men (infantry + cavalry) is wiped from the map instead of retreating, and
# its commander returns to the available pool. Stops tiny remnants drifting around.
MIN_ARMY_SIZE = 100


# ── Historical 1808 starting positions ───────────────────────────────────────
# France enters from the Pyrenees through five passes / border corridors.
# All other nodes default to ALLIES (Spanish / Portuguese territory).
FRANCE_START_NODES = ['LJQ', 'VER', 'MAY', 'PAM', 'HUE']
# LJQ=La Jonquera (French border / Catalan road from Perpignan), VER=Verdun (central Pyrenean pass),
# MAY=Maya (western pass / Navarre), PAM=Pamplona, HUE=Huesca (Aragón)

# Allies spread across Portugal and Spain.  8 armies ≤ MAX_ARMIES cap.
ALLIES_START_NODES = ['LIS', 'CAD', 'MAD', 'ZAR', 'BAR', 'SEV', 'OPO', 'STG']
# LIS=Lisbon, CAD=Cádiz, MAD=Madrid, ZAR=Zaragoza, BAR=Barcelona, SEV=Sevilla, OPO=Oporto, STG=Santiago


INITIAL_ARMY_SIZE = 30000

# Per-army troop sizes (infantry+cavalry headcount before split)
FRANCE_START_SIZES = {
    'LJQ': 17_000,   # Saint-Cyr's 7th Corps — entering via La Jonquera / Perpignan road (~17k per Oman)
    'VER': 50_000,   # Ney's central column: Guard + 4th Corps (~50k)
    'MAY': 25_000,   # Soult's 2nd Corps — Bayonne / western pass (~25k)
    'PAM': 18_000,   # Napoleon — Pamplona forward base (~18k); departs turn 7
    'HUE': 22_000,   # Victor's 1st Corps — Aragon (~22k)
}
ALLIES_START_SIZES = {
    'LIS': 25_000,   # Silveira — Portuguese regulars defending Lisbon
    'CAD': 18_000,   # Castaños — Army of Andalusia (reduced post-Bailén/Tudela)
    'MAD': 22_000,   # Cuesta — Army of Estremadura (~15–20k + Madrid garrison)
    'ZAR': 30_000,   # Morillo — Zaragoza defenders (~30k at the 2nd siege)
    'BAR': 20_000,   # Moore — British contingent in Catalonia
    'SEV': 22_000,   # Blake — Andalusian / southern Spanish army
    'OPO': 15_000,   # Freire — northern Portuguese
    'STG': 12_000,   # La Romana — Galician army (returning from Denmark, ~10–12k)
}

# Starting sub-factions for Allied armies
ALLIES_START_SUBFACTIONS = {
    'LIS': SUBFACTION_PORTUGUESE,
    'CAD': SUBFACTION_SPANISH,
    'MAD': SUBFACTION_SPANISH,
    'ZAR': SUBFACTION_SPANISH,
    'BAR': SUBFACTION_BRITISH,
    'SEV': SUBFACTION_SPANISH,
    'OPO': SUBFACTION_PORTUGUESE,
    'STG': SUBFACTION_SPANISH,
}

# Relative map file paths
NODES_FILE = 'Map/nodes.csv'
EDGES_FILE = 'Map/edges.csv'

# Cavalry and artillery ratios (faction / sub-faction)
# Used to derive attacker_cav / attacker_arty for the battle model.
CAVALRY_RATIO = {
    FRANCE: 0.15,
    SUBFACTION_BRITISH: 0.12,
    SUBFACTION_SPANISH: 0.10,
    SUBFACTION_PORTUGUESE: 0.11,
}
ARTY_PER_1000 = {               # Artillery guns per 1,000 troops
    FRANCE: 3.0,   
    SUBFACTION_BRITISH: 2.5,
    SUBFACTION_SPANISH: 1.5,
    SUBFACTION_PORTUGUESE: 2.0,
}

# Technology advantage table 
# techa: attacker's relative technological advantage (0=parity or defender's, 1=slight adv.)
# Key: (attacker_faction, defender_subfaction)
# France vs Iberian allies has techa=1; all other matchups techa=0.
TECHA_TABLE = {
    (FRANCE, SUBFACTION_BRITISH):    0,
    (FRANCE, SUBFACTION_SPANISH):    1,
    (FRANCE, SUBFACTION_PORTUGUESE): 1,
    (ALLIES, SUBFACTION_NONE):       0,   # Allies attacking France: parity
}

# ── Morale advantage table
# morala: attacker's relative morale advantage (0=parity or defender's, 1=slight adv.)
# Key: (attacker_subfaction, defender_subfaction)
# Iberian allies vs France has morala=1; all other matchups morala=0.
MORALA_TABLE = {
    (SUBFACTION_BRITISH, FRANCE): 0,
    (SUBFACTION_SPANISH, FRANCE): 1,
    (SUBFACTION_PORTUGUESE, FRANCE): 1,
    (SUBFACTION_NONE, SUBFACTION_BRITISH): 0,   # France attacking Allies: parity
    (SUBFACTION_NONE, SUBFACTION_PORTUGUESE): 0,
    (SUBFACTION_NONE, SUBFACTION_PORTUGUESE): 0
}

# Battle history windows
MOMNTA_WINDOW = 3
INITA_WINDOW = 5

# Supply depot node types (used in BFS logsa calculation)
SUPPLY_DEPOT_TYPES = {'capital', 'regional_capital', 'major_city', 'city'}
LOGSA_THRESHOLDS = (100, 250)  # km diff thresholds for logsa=1 and logsa=2


LOPSIDED_RATIO = 8.0   # Strength ratio above which lopsided battles override kicks in
LOPSIDED_WINNER_CAS_RATE = 0.05  # 5% -> victorious side takes light losses
LOPSIDED_LOSER_CAS_RATE = 0.40  # 40% -> routed side takes heavy losses

# Garrison constants
SIEGE_GARRISON_LOSS_RATE = 0.30   # Fraction of garrison lost per siege turn
SIEGE_ATTACKER_LOSS_RATE = 0.05   # Fraction of current garrison lost by attacker per siege turn
MAX_GARRISON_OBS = 3000   # Normalisation denominator for observation

# Commander quality lookup
# Float in [0, 1] representing historical win rate / command quality.
# Used in the observation vector so agents learn commander value.
# Dataset global mean ≈ 0.565  ->  used as fallback for unknown commanders.
# Sources: History_Battles.xlsx stats where available (Wellington 6/6=1.0)
# remaining values from Peninsular War historical record
COMMANDER_QUALITY = {
    # French commanders
    'NAPOLEON': 0.92,   # Best commander of the war
    'LANNES': 0.82,   # Napoleon's best Marshal in Spain; decisive at Zaragoza; dies May 1809
    'SOULT': 0.57,   # Competent; Coruña win, Albuera loss
    'BESSIÈRES': 0.57,   # Reliable cavalry; administrative in north Iberia
    'LEFEBVRE': 0.54,   # Competent at sieges (Zaragoza); limited in open field
    'NEY': 0.60,   # Excellent rearguard; mixed offensive
    'MASSENA': 0.50,   # Good commander with a mixed record in Iberia
    'MARMONT': 0.53,   # Solid until Salamanca disaster
    'VICTOR': 0.55,   # Ucles/Medellin wins; lost Talavera
    'SUCHET': 0.70,   # Most successful French commander in Iberia
    'SAINT-CYR': 0.63,   # Solid in Catalonia; won at Valls
    'AUGEREAU': 0.46,   # Ill and underperforming in Catalonia
    'MACDONALD': 0.61,   # Reliable; steadied Catalonia after Augereau
    'JUNOT': 0.47,   # Lost Vimeiro; limited ability
    'JOSEPH BONAPARTE': 0.40,  # Poor commander; lost Vitoria
    'CLAUSEL': 0.55,   # Reliable but often in hard situations
    # British commanders 
    'WELLINGTON': 0.90,   # Best allied commander
    'BERESFORD': 0.55,   # Won Albuera at high cost
    'HILL': 0.65,   # Consistently reliable; few defeats
    'MOORE': 0.58,   # Coruña rearguard; capable general
    'GRAHAM': 0.60,   # Won Barrosa; dependable
    'PICTON': 0.65,   # Aggressive 3rd Division commander; excellent record
    # Spanish 
    'CASTANOS': 0.52,   # Won Bailén; mixed thereafter
    'CUESTA': 0.38,   # Medellin disaster; poor record
    'BLAKE': 0.35,   # Repeatedly defeated by French
    'LA ROMANA': 0.45,   # Struggled throughout
    'MORILLO': 0.55,   # Reliable; worked well with Wellington mid-late war
    # Portuguese 
    'SILVEIRA': 0.55,   # Defended N. Portugal, recaptured Chaves
    'FREIRE': 0.55,   # Solid divisional commander throughout
}
COMMANDER_QUALITY_DEFAULT = 0.565   # global mean -> fallback for unknowns

# Commander seniority / chain of command
COMMANDER_SENIORITY = {
    # French chain of command 
    'NAPOLEON': 1,   # Emperor; supreme commander
    'JOSEPH BONAPARTE': 2,   # King of Spain; nominal Iberian CiC
    'LANNES': 3,   # First Marshal
    'SOULT': 4,   # Senior Marshal — Bayonne to Lisbon
    'MASSENA': 5,   # Senior Marshal
    'NEY': 6,   # Senior Marshal
    'BESSIÈRES': 7,   # Senior Marshal — Imperial Guard cavalry
    'AUGEREAU': 8,   # Senior Marshal — Catalonia
    'SUCHET': 9,   # Marshal - Most succesful in Iberia
    'LEFEBVRE': 10,   # Marshal — Zaragoza siege
    'VICTOR': 11,   # Marshal — Aragon & Talavera
    'MACDONALD': 12,   # Marshal — Catalonia
    'SAINT-CYR': 13,   # Marshal — Catalonia
    'MARMONT': 14,   # Marshal — Army of Portugal
    'JUNOT': 15,   # General of Division — Sintra
    'CLAUSEL': 16,   # General of Division — late-war replacement
    # Allied chain of command
    'WELLINGTON': 1,   # Viscount/Duke; supreme Allied CiC from turn 22
    'MOORE': 2,   # Lieutenant-General; British CiC before Wellington
    'BERESFORD': 3,   # Marshal of Portugal; reformed Portuguese army
    'HILL': 4,   # Lieutenant-General; Wellington's trusted second
    'GRAHAM': 5,   # Lieutenant-General; Barrosa
    'PICTON': 6,   # Major-General; 3rd Division
    'CASTANOS': 7,   # Captain-General; hero of Bailén
    'SILVEIRA': 8,   # General; northern Portugal
    'FREIRE': 9,   # General; Portuguese division
    'LA ROMANA': 10,   # Captain-General; Galician army
    'MORILLO': 11,   # General; Spanish division
    'CUESTA': 12,   # Captain-General; Army of Extremadura
    'BLAKE': 13,   # Captain-General; Army of Galicia / Valencia
}
COMMANDER_SENIORITY_DEFAULT = 999   # Fallback for UNKNOWN

# Starting commanders
FRANCE_START_COMMANDERS = {
    'LJQ': 'LANNES',    # NE / Catalonia entry, departs turn 16
    'VER': 'NEY',       # main pass -> Pamplona–Burgos road (50k corps)
    'MAY': 'SOULT',     # western Pyrenees -> Galicia / Portugal
    'PAM': 'NAPOLEON',  # Pamplona forward base, departs turn 7
    'HUE': 'VICTOR',    # Aragon foothills -> Zaragoza support
}
ALLIES_START_COMMANDERS = {
    'LIS': 'SILVEIRA',    # Portuguese army defending Lisbon
    'CAD': 'CASTANOS',    # southern Spanish army, hero of Bailén
    'MAD': 'CUESTA',      # Castilian army at Madrid
    'ZAR': 'MORILLO',     # Spanish Army of Aragon (Zaragoza)
    'BAR': 'MOORE',       # British contingent in Catalonia
    'SEV': 'BLAKE',       # Andalusian / Army of the North
    'OPO': 'FREIRE',      # northern Portuguese
    'STG': 'LA ROMANA',   # Galician army
}

# Starting Commander pools
# Starting commanders are assigned at reset; the rest sit in the available pool.
FRANCE_COMMANDER_POOL = [
    'NAPOLEON', 'LANNES', 'SOULT', 'BESSIÈRES', 'LEFEBVRE',
    'NEY', 'MASSENA', 'MARMONT', 'VICTOR', 'SUCHET',
    'SAINT-CYR', 'AUGEREAU', 'MACDONALD', 'JUNOT',
    'JOSEPH BONAPARTE', 'CLAUSEL',
]
ALLIES_COMMANDER_POOL = [
    # WELLINGTON excluded — he arrives via historical event at turn 22
    # MOORE, CASTANOS, CUESTA, BLAKE, MORILLO, SILVEIRA, FREIRE, LA ROMANA
    # are all assigned at start; remaining pool used for splits / replacements
    'BERESFORD', 'HILL', 'GRAHAM', 'PICTON',
    'CASTANOS', 'CUESTA', 'BLAKE', 'LA ROMANA', 'MORILLO',
    'SILVEIRA', 'FREIRE', 'MOORE',
]

# Reinforcement depots
# Two depots per faction — rate in the schedule is MEN PER DEPOT PER TURN,
# so both factions receive the same number of depot accumulations each turn.
#
# France: VER (northern road via Bayonne/Roncesvalles) +
#         LJQ (Mediterranean / Catalan road via Perpignan — La Jonquera border crossing).
# Allies: LIS (British sea-supply, Wellington's base) +
#         CAD (Spanish southern heartland, never fell to France).
FRANCE_REINF_DEPOTS = ['VER', 'LJQ']
ALLIES_REINF_DEPOTS = ['LIS', 'CAD']

# Sub-faction for troops spawned at each Allied depot
ALLIES_REINF_SUBFACTIONS = {
    'LIS': SUBFACTION_BRITISH,   # Wellington's base; primary British landing point
    'CAD': SUBFACTION_SPANISH,   # Spanish southern heartland
}

# Pool mechanics
REINF_SPAWN_THRESHOLD = 20000   # Pool must reach this to spawn a new army
REINF_POOL_CAP = 40000   # Hard ceiling per depot (2x threshold); excess lost

# Reinforcement schedule
# (turn_start, rate_per_depot_france, rate_per_depot_allies)
# Rate = men added to EACH depot's pool per turn (same depot count per side).
# France totals = rate x 2;  Allies totals = rate x 2.
# Periods are derived from Wikipedia strength data and historical events:
#   turn  0  = November 1808  (simulation start)
#   turn 65  = April 1809     (Fifth Coalition begins)
#   turn 90  = October 1809   (Wagram; Austria makes peace)
#   turn 165 = mid-1811       (Russian campaign preparations)
#   turn 209 = mid-1812       (Grande Armee crosses into Russia)
#   turn 260 = early 1813     (France fighting for survival in Germany)

REINFORCEMENT_SCHEDULE = [
    (  0, 750, 400),   # Napoleon's surge; Allies disorganised
    ( 65, 300, 500),   # Fifth Coalition diverts French strength
    ( 90, 400, 700),   # France recovers post-Wagram; Portuguese reform bearing fruit
    (165, 200, 600),   # Russian prep strips French veterans; Allied strength growing
    (209, 75, 900),   # Russian campaign; Wellington on the offensive
    (260, 50, 800),   # France fighting for survival; Allied peak
]

# March attrition (road-type penalty)
# Armies that exceed the threshold lose a fraction of their men during a move.
# Artillery is not affected.
# Primary roads: no penalty for any army.
# Secondary roads: armies above MARCH_SECONDARY_THRESHOLD lose MARCH_SECONDARY_RATE.
# Tertiary roads: armies above MARCH_TERTIARY_THRESHOLD lose MARCH_TERTIARY_RATE.
MARCH_SECONDARY_THRESHOLD = 30_000   # Men (infantry + cavalry)
MARCH_SECONDARY_RATE = 0.02     # 2% loss
MARCH_TERTIARY_THRESHOLD = 10_000   # Men (infantry + cavalry)
MARCH_TERTIARY_RATE = 0.03     # 3% loss

# Casualty split ratios
CAVALRY_CASUALTY_RATIO = 0.10

# Reward weights
W_NODE_DELTA = 0.02   
W_CASUALTY = 0.00005
W_BATTLE_OUTCOME = 0.03
W_TERRITORY_CONSOL = 0.05    
W_ARMY_CONSOL = 0.03
W_ENEMY_DEPOT = 0.05    
W_DEPOT_BASE_PENALTY = 0.20
W_DEPOT_SPIKE = 0.40

# Node capture value
# Reward for gaining (or, negated, losing) a node
NODE_CAPTURE_VALUE = {
    'capital': 30.0,
    'regional_capital': 15.0,
    'major_city': 8.0,
    'city': 3.0,
    'town': 1.0,
    'intersection': 0.0,
}

# Commander death probability
COMMANDER_DEATH_PROB = 0.02

# Historical commander events: (turn, faction, event_type, commander_name)
HISTORICAL_COMMANDER_EVENTS = [
    ( 7, FRANCE, 'remove', 'NAPOLEON'),
    (16, FRANCE, 'remove', 'LANNES'),
    (22, ALLIES, 'add', 'WELLINGTON'),
]
