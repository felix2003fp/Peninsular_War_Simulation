"""
renderer.py
-----------
Live map visualisation for the Peninsular War simulation.

Displays the topographic map with:
  - Nodes coloured by faction owner  (blue=France, crimson=Allies, grey=Neutral)
  - Node size scaled by strategic importance (same scheme as visualise_graph.py)
  - Real nation flags (PNG from Map/) drawn at every army position
  - Troop-count label beneath each flag
  - Scoreboard overlay (turn, troops, nodes held)
  - Battle markers (crossed swords ✕) on nodes that fought this turn

Call renderer.update() after every env.step(). The window refreshes
without blocking the terminal input loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

if TYPE_CHECKING:
    from .peninsular_war_env import PeninsularWarEnv

from .config import (
    FRANCE, ALLIES, NEUTRAL, MAX_TURNS,
    SUBFACTION_BRITISH, SUBFACTION_SPANISH, SUBFACTION_PORTUGUESE,
)

# ── Coordinate calibration (mirrors visualise_graph.py) ──────────────────────

# Coordinate projection (lat/lon -> native topo-map pixel) is shared with
# Env/pygame_renderer.py and Map/visualise_graph.py — defined once in
# Map/map_projection.py. Ensure the project root is importable, then alias.
import sys as _sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from Map.map_projection import latlon_to_pixel as _to_px


# ── Flag images (loaded from Map/ PNG files) ──────────────────────────────────

def _load_flag(filename: str) -> np.ndarray:
    """
    Load a flag image from Map/ using Pillow (handles PNG, WebP, JPEG, etc.).
    Returns a uint8 RGB numpy array.
    """
    from PIL import Image
    path = Path(__file__).parent.parent / 'Map' / filename
    img  = Image.open(str(path)).convert('RGB').resize((36, 24), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


_FRANCE_FLAG   = _load_flag('Flags/French_Flag.png')
_UK_FLAG       = _load_flag('Flags/British_Flag.png')
_SPAIN_FLAG    = _load_flag('Flags/Spanish_Flag.png')
_PORTUGAL_FLAG = _load_flag('Flags/Portuguese_Flag.png')


# ── Node symbol images (mirrors Map/visualise_graph.py) ───────────────────────

def _load_node_image(filename: str) -> np.ndarray:
    """Load a node-symbol PNG from Map/ at native resolution, RGBA."""
    from PIL import Image
    path = Path(__file__).parent.parent / 'Map' / filename
    return np.array(Image.open(str(path)).convert('RGBA'))


# node_type → symbol filename (shared, defined in Map/map_projection.py)
from Map.map_projection import (
    NODE_SYMBOL_FILE as _NODE_IMAGE_FILE,
    REINF_DEPOT_FILE as _REINF_DEPOT_FILE,
)

# node_id → symbol filename overrides
# Only depots override the symbol; all other nodes follow their node_type so
# Cádiz (regional_capital) and Lisboa (capital) render with the correct symbols.
_NODE_IMAGE_OVERRIDE = {
    'VER': _REINF_DEPOT_FILE,                       # Verdun       → reinforcement depot
    'LJQ': _REINF_DEPOT_FILE,                       # La Jonquera  → reinforcement depot
}

# node_type → on-map zoom factor for its symbol image
_NODE_ZOOM = {
    'capital':          0.096,
    'regional_capital': 0.082,
    'major_city':       0.068,
    'city':             0.054,
    'town':             0.045,
    'intersection':     0.035,
}
_REINF_DEPOT_ZOOM = 0.077

# node_id → zoom override (paired with _NODE_IMAGE_OVERRIDE)
_NODE_ZOOM_OVERRIDE = {
    'VER': _REINF_DEPOT_ZOOM,
    'LJQ': _REINF_DEPOT_ZOOM,
}

_NODE_IMAGE_CACHE: dict[str, np.ndarray] = {}
_NODE_TINT_CACHE: dict[tuple, np.ndarray] = {}

# Owner → symbol tint colour (RGB). France blue, Allies red, Neutral grey.
_SYMBOL_TINT = {FRANCE: (65, 105, 225), ALLIES: (196, 30, 58), NEUTRAL: (200, 200, 200)}


def _tint(arr: np.ndarray, rgb) -> np.ndarray:
    """Recolour an RGBA symbol to `rgb`, preserving its shading and transparency.
    The brightest non-transparent pixel maps to full `rgb`; darker pixels scale
    down, so the symbol stays recognisable for light or dark source art."""
    out = arr.copy()
    alpha = out[..., 3] if out.shape[2] == 4 else np.full(out.shape[:2], 255)
    lum = out[..., :3].astype(float).mean(axis=2)
    mask = alpha > 0
    mx = lum[mask].max() if mask.any() else 255.0
    v = (lum / (mx or 1.0)).clip(0, 1)[..., None]
    out[..., :3] = (np.array(rgb, float).reshape(1, 1, 3) * v).clip(0, 255).astype(np.uint8)
    return out


def _node_symbol(node_id: str, ntype: str, owner: int | None = None):
    """Return (image array, zoom factor). If `owner` is given, the symbol is
    tinted by faction (France blue / Allies red / Neutral grey)."""
    filename = _NODE_IMAGE_OVERRIDE.get(node_id, _NODE_IMAGE_FILE.get(ntype, _NODE_IMAGE_FILE['town']))
    zoom     = _NODE_ZOOM_OVERRIDE.get(node_id, _NODE_ZOOM.get(ntype, _NODE_ZOOM['town']))
    if filename not in _NODE_IMAGE_CACHE:
        _NODE_IMAGE_CACHE[filename] = _load_node_image(filename)
    base = _NODE_IMAGE_CACHE[filename]
    if owner is None or owner not in _SYMBOL_TINT:
        return base, zoom
    key = (filename, owner)
    if key not in _NODE_TINT_CACHE:
        _NODE_TINT_CACHE[key] = _tint(base, _SYMBOL_TINT[owner])
    return _NODE_TINT_CACHE[key], zoom

# ── Style tables ──────────────────────────────────────────────────────────────

_NODE_FC = {FRANCE: '#4169E1', ALLIES: '#C41E3A', NEUTRAL: '#CCCCCC'}
_NODE_EC = {FRANCE: '#1A3A8F', ALLIES: '#7D0000', NEUTRAL: '#888888'}

_NODE_SIZE = {
    'capital':          300,
    'regional_capital': 180,
    'major_city':       100,
    'city':              55,
    'town':              24,
    'intersection':      12,
}

_EDGE_STYLE = {
    'primary':   ('black',   2.0, 0.90, '-'),
    'secondary': ('#555555', 1.2, 0.75, '--'),
    'tertiary':  ('#555555', 1.0, 0.65, ':'),
}


# ── Renderer class ────────────────────────────────────────────────────────────

class MapRenderer:
    """
    Initialise once after env.reset(), then call update(battle_log) each turn.

    Parameters
    ----------
    env : PeninsularWarEnv
    """

    def __init__(self, env: 'PeninsularWarEnv'):
        self.env = env
        root     = Path(__file__).parent.parent

        # ── Static node/edge metadata ─────────────────────────────────────────
        nodes_df = pd.read_csv(root / 'Map' / 'nodes.csv')
        edges_df = pd.read_csv(root / 'Map' / 'edges.csv')

        self._ntype: dict[str, str] = (
            nodes_df.set_index('node_id')['node_type'].to_dict()
            if 'node_type' in nodes_df.columns
            else {nid: 'town' for nid in env.node_ids}
        )
        self._rtype: dict[tuple, str] = {}
        for _, row in edges_df.iterrows():
            rt = row.get('road_type', 'secondary')
            self._rtype[(row['node1'], row['node2'])] = rt
            self._rtype[(row['node2'], row['node1'])] = rt

        # ── Pre-compute pixel positions for every node ────────────────────────
        self._px: dict[str, float] = {}
        self._py: dict[str, float] = {}
        for nid in env.node_ids:
            px, py         = _to_px(env._lat[nid], env._lon[nid])
            self._px[nid]  = px
            self._py[nid]  = py

        # ── Set up figure ─────────────────────────────────────────────────────
        plt.ion()
        map_path = root / 'Map' / 'Iber_Pen_Topo_Map.jpg'
        if map_path.exists():
            self._bg = mpimg.imread(str(map_path))
            self._ih, self._iw = self._bg.shape[:2]
        else:
            print('[renderer] Map image not found — using plain background.')
            self._bg = None
            self._iw, self._ih = 2200, 1800

        fig_w  = 15
        fig_h  = fig_w * self._ih / self._iw
        self.fig, self.ax = plt.subplots(figsize=(fig_w, fig_h))
        self.fig.patch.set_facecolor('black')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.axis('off')
        self.ax.set_xlim(0, self._iw)
        self.ax.set_ylim(self._ih, 0)   # image y-axis: 0 at top

        # ── Draw permanent background (map + edges) ───────────────────────────
        if self._bg is not None:
            self.ax.imshow(self._bg, extent=[0, self._iw, self._ih, 0], zorder=0)

        for _, row in edges_df.iterrows():
            n1, n2 = row['node1'], row['node2']
            if n1 not in self._px or n2 not in self._px:
                continue
            rt                = row.get('road_type', 'secondary')
            col, lw, alp, ls  = _EDGE_STYLE.get(rt, _EDGE_STYLE['secondary'])
            self.ax.plot(
                [self._px[n1], self._px[n2]],
                [self._py[n1], self._py[n2]],
                color=col, linewidth=lw, alpha=alp, linestyle=ls,
                zorder=1, solid_capstyle='round',
            )

        # Container for dynamic artists (cleared each update)
        self._dyn: list = []

        self.update([])
        plt.show(block=False)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, battle_log: List[dict] | None = None):
        """
        Redraw all dynamic elements.  Call after every env.step().

        Parameters
        ----------
        battle_log : list of battle dicts from env.step() info['battles']
        """
        for art in self._dyn:
            try:
                art.remove()
            except Exception:
                pass
        self._dyn.clear()

        self._draw_nodes()
        self._draw_flags()
        if battle_log:
            self._draw_battle_markers(battle_log)
        self._draw_scoreboard()

        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    # ── Internal draw methods ─────────────────────────────────────────────────

    def _draw_nodes(self):
        env = self.env
        for i, nid in enumerate(env.node_ids):
            owner  = int(env.owner[i])
            ntype  = self._ntype.get(nid, 'town')
            px, py = self._px[nid], self._py[nid]

            # Node symbol (PNG), tinted by faction owner
            img_arr, zoom = _node_symbol(nid, ntype, owner)
            imbox = OffsetImage(img_arr, zoom=zoom)
            ab = AnnotationBbox(imbox, (px, py), frameon=False, zorder=3,
                                box_alignment=(0.5, 0.5))
            self.ax.add_artist(ab)
            self._dyn.append(ab)

            # Node ID label (skip intersections)
            if ntype != 'intersection':
                txt = self.ax.text(
                    px, py + 10, nid,
                    fontsize=4.5, color='white', weight='bold',
                    ha='center', va='top', zorder=4,
                    bbox=dict(boxstyle='round,pad=0.1', fc='#000000',
                              ec='none', alpha=0.55),
                )
                self._dyn.append(txt)

    def _draw_flags(self):
        env = self.env
        _allied_flag = {
            SUBFACTION_BRITISH:    _UK_FLAG,
            SUBFACTION_SPANISH:    _SPAIN_FLAG,
            SUBFACTION_PORTUGUESE: _PORTUGAL_FLAG,
        }
        for i, nid in enumerate(env.node_ids):
            f_men  = int(env.france_infantry[i] + env.france_cavalry[i])
            f_arty = int(env.france_artillery[i])
            a_men  = int(env.allies_infantry[i]  + env.allies_cavalry[i])
            a_arty = int(env.allies_artillery[i])
            px, py = self._px[nid], self._py[nid]

            if f_men > 0 or f_arty > 0:
                self._add_flag(_FRANCE_FLAG, px - 8, py - 14, zoom=0.5)
                self._add_label(px - 8, py - 14, f_men, '#4169E1')
            if a_men > 0 or a_arty > 0:
                sf   = int(env.sub_faction[i])
                flag = _allied_flag.get(sf, _UK_FLAG)
                offset = 10 if (f_men > 0 or f_arty > 0) else 0
                self._add_flag(flag, px + offset, py - 14, zoom=0.5)
                self._add_label(px + offset, py - 14, a_men, '#C41E3A')

    def _add_flag(self, img: np.ndarray, x: float, y: float, zoom: float):
        im = OffsetImage(img, zoom=zoom, resample=True)
        ab = AnnotationBbox(im, (x, y), frameon=False, zorder=5,
                            box_alignment=(0.5, 0.0))
        self.ax.add_artist(ab)
        self._dyn.append(ab)

    def _add_label(self, x: float, y: float, men: int, bg: str):
        # (x, y) is the flag's bottom-centre anchor. Show only the men count
        # (infantry + cavalry; artillery is omitted) and place it just to the
        # RIGHT of the flag, vertically centred on it. The position is given as
        # an offset in points so it stays correct regardless of the data zoom.
        label = f'{men // 1000}k' if men >= 1000 else str(men)
        txt = self.ax.annotate(
            label, xy=(x, y), xytext=(12, 6), textcoords='offset points',
            fontsize=5.5, color='white', weight='bold',
            ha='left', va='center', zorder=6,
            bbox=dict(boxstyle='round,pad=0.15', fc=bg, ec='none', alpha=0.80),
        )
        self._dyn.append(txt)

    def _draw_battle_markers(self, battle_log: List[dict]):
        """Mark nodes where a battle occurred this turn with a red ✕."""
        for b in battle_log:
            nid = b['node']
            if nid not in self._px:
                continue
            txt = self.ax.text(
                self._px[nid], self._py[nid] + 16, '✕',
                fontsize=11, color='red', weight='bold',
                ha='center', va='top', zorder=7,
            )
            self._dyn.append(txt)

    def _draw_scoreboard(self):
        env  = self.env
        f_t  = int(np.sum(env.france_infantry + env.france_cavalry))
        a_t  = int(np.sum(env.allies_infantry  + env.allies_cavalry))
        f_g  = int(np.sum(env.france_artillery))
        a_g  = int(np.sum(env.allies_artillery))
        f_n  = int(np.sum(env.owner == FRANCE))
        a_n  = int(np.sum(env.owner == ALLIES))
        t    = env.turn

        board = (
            f"  Turn {t:>3} / {MAX_TURNS}  ({t / 52:.1f} yrs)\n"
            f"  France  {f_t:>8,} men  {f_g:>4} guns  {f_n:>3} nodes\n"
            f"  Allies  {a_t:>8,} men  {a_g:>4} guns  {a_n:>3} nodes  "
        )
        txt = self.ax.text(
            self._iw - 14, self._ih - 14, board,
            fontsize=8.5, color='white', family='monospace',
            ha='right', va='bottom', zorder=8,
            bbox=dict(boxstyle='round,pad=0.5', fc='#111111',
                      ec='#555555', alpha=0.82),
        )
        self._dyn.append(txt)
