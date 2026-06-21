# Peninsular War — Multi-Agent Reinforcement Learning Simulation

A discrete-time, agent-based simulation of the Peninsular War (1808–1814) in which
autonomous agents representing the **French** and the **Allied** (British, Spanish and
Portuguese) forces learn strategic military decision-making through Multi-Agent
Reinforcement Learning. The Iberian Peninsula is modelled as a weighted graph of
historically grounded nodes connected by a period road network; battles are resolved by
a supervised model trained on historical engagements; and documented historical events
(commander arrivals, reinforcement schedules) are injected into the
timeline. Agents are trained with `MaskablePPO` under self-play and evaluated by
comparing their simulated territorial-control trajectories against the historical record.

For the full description of the methodology, the experiments, and the **conclusions**, see
the written report (`TFG_FelixFernandezPeñafiel.pdf`) and the `results/` folder
(in particular `results/analysis/REPORT.md`).

---

## Repository structure

### Top-level scripts

| File | What it does |
|------|--------------|
| `train.py` | The training pipeline. Runs the self-play / iterated-best-response phases for either tier (Tier 1 = joint action, Tier 2 = per-army local action) using `MaskablePPO`. |
| `play.py` | Play the game yourself against a trained model. Choose your side with `--play-faction` and the opponent's tier with `--tier`. |
| `watch_pygame.py` | Watch two trained agents play each other, rendered with Pygame. Can record an `.mp4` (`--record`), reproduce a specific ensemble episode (`--reproduce --seed`), or just save the turn-0 map (`--snapshot`). |
| `simulate_history.py` | Runs large self-play ensembles (e.g. 500 games/tier) and records, for every run and week, which faction held each node. Writes the tensors/CSVs in `results/`. |
| `analyze_history.py` | Compares the simulated ensembles in `results/` against `historical_truth.csv` and produces the figures and metrics in `results/analysis/`. |
| `historical_truth.csv` | Week-by-week ground truth (1 = French, 0 = Allied) of the ten key cities used for the historical comparison. |

### `Env/` — the simulation environment

| File | What it does |
|------|--------------|
| `peninsular_war_env.py` | The core two-faction `gymnasium.Env`: state, transition dynamics, battles, sieges, captures, reinforcements, observation/action spaces, reward. |
| `config.py` | All constants: factions, map files, reward weights, reinforcement schedules, commander pools, historical-event timetable, etc. |
| `battle_model.py` | Loads the trained battle-outcome models and resolves each engagement (winner + casualties), with a deterministic override for lopsided battles. |
| `sb3_wrapper.py` | Single-agent Stable-Baselines3 wrappers around the two-faction env: action masking, the Tier-1 (full-map) and Tier-2 (local) observation/action handling, and the frozen-opponent wrappers used for self-play. |
| `renderer.py` | Matplotlib live map renderer (used by `play.py`). |
| `pygame_renderer.py` | Pygame map renderer with `.mp4`/PNG recording (used by `watch_pygame.py`). |

### `Map/` — the graph and visual assets

| File | What it does |
|------|--------------|
| `nodes.csv` | The 230 map nodes (coordinates, type, terrain, garrison, strategic importance). |
| `edges.csv` | The 433 road connections (distance, terrain, road class). |
| `map_projection.py` | Single source of truth for the lat/lon → map-pixel projection and the node-type → symbol-image map, shared by the renderers and `visualize_graph.py`. |
| `visualize_graph.py` | Renders the static charted map from `nodes.csv` / `edges.csv`. |
| `Iber_Pen_Topo_Map.jpg` | Base topographic map image used for rendering. |
| `Charted_Map.png` | The final rendered map of the simulation. |
| `Flags/` | National flag images (France, Britain, Spain, Portugal). |
| `Node_Symbols/` | Per-node-type symbol images (capital, city, town, depot, …). |

### `Battle_Outcome_Model/` — the battle predictor

| File | What it does |
|------|--------------|
| `battle_outcome_wina.ipynb` | Notebook that trains the **winner** prediction model. |
| `battle_outcome_casualties.ipynb` | Notebook that trains the **casualty** (men + artillery) regression models. |
| `History_Battles.xlsx` | The historical-battles dataset the models are trained on. |

### `results/` — simulation outputs and analysis

| Item | What it is |
|------|------------|
| `episodes_tier{1,2}.csv` | One row per simulated game: winner, end reason, end turn, final node counts. |
| `analysis/REPORT.md` | Headline comparison numbers (Brier, agreement, capture-order correlation, win rates) for both tiers. |
| `analysis/city_metrics.csv` | Per-city fidelity metrics. |
| `analysis/*.png` | The comparison figures: `prob_curves`, `fidelity_over_time`, `time_to_capture`, `outcomes`. |

### Report

| Item | What it is |
|------|------------|
| `TFG_FelixFernandezPeñafiel.pdf` | The full written report (methodology, experiments, results, conclusions). |

---

## Not included in the repository

Some large or regenerable artifacts are intentionally excluded (see `.gitignore`):

- **`models/`** — the trained RL checkpoints (tens of GB). Without them, `play.py`,
  `watch_pygame.py` and `simulate_history.py` will report "model not found"; they can be
  produced by running `train.py`.
- **`Battle_Outcome_Model/*.pkl`** — the serialized battle-outcome models. Regenerate them
  by running the two notebooks in `Battle_Outcome_Model/`.
- **`wandb/`** — Weights & Biases training logs.
- **`Map/enrich_map.py`** and the local Python virtual environment.

---

## Requirements

Python 3.11+ with: `gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch` (CPU),
`numpy`, `pandas`, `matplotlib`, `pygame`, `scikit-learn`, `imageio` (for recording),
and `wandb` (optional, for training logs).

## Quick start

```bash
# Train (CPU); see train.py for the tier / phase options
python train.py

# Play against a trained model (you = France vs the Tier-2 Allies model)
python play.py --play-faction france --tier 2

# Watch two trained agents and record the game
python watch_pygame.py --record game.mp4

# Reproduce the historical comparison
python simulate_history.py --tiers 1 2 --n-sims 500
python analyze_history.py
```

## Important

After running **`train.py`**, remember to update the model paths in **`watch_pygame.py`**, **`simulate_history.py`** and **`play.py`**. Right now they are coded with the paths for my models, and of course will fail if not updated to yours.
