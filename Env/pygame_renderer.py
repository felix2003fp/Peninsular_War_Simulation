from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
import numpy as np
import pandas as pd
import pygame
import sys as _sys

if TYPE_CHECKING:
    from .peninsular_war_env import PeninsularWarEnv

from .config import (
    FRANCE, ALLIES, NEUTRAL, MAX_TURNS,
    SUBFACTION_BRITISH, SUBFACTION_SPANISH, SUBFACTION_PORTUGUESE,
)

# Coordinate calibration
# The lat/lon -> native-pixel projection is shared with Env/renderer.py and
# Map/visualise_graph.py, defined once in Map/map_projection.py.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from Map.map_projection import latlon_to_pixel as _to_native_px


# Style tables
_NODE_FC = {FRANCE: (65, 105, 225), ALLIES: (196, 30, 58), NEUTRAL: (204, 204, 204)}

# Node-symbol PNG zoom factor (fraction of native symbol size) by node type
_NODE_ZOOM = {
    'capital': 0.096,
    'regional_capital': 0.082,
    'major_city': 0.068,
    'city': 0.054,
    'town': 0.045,
    'intersection': 0.035,
}
_REINF_DEPOT_ZOOM = 0.077

# Node_type -> symbol filename (shared, defined in Map/map_projection.py)
from Map.map_projection import (
    NODE_SYMBOL_FILE as _NODE_IMAGE_FILE,
    REINF_DEPOT_FILE as _REINF_DEPOT_FILE,
)

# Only depots override the symbol; all other nodes follow their node_type so
# Cadiz (regional_capital) and Lisboa (capital) render with the correct symbols.
_NODE_IMAGE_OVERRIDE = {
    'VER': _REINF_DEPOT_FILE,
    'LJQ': _REINF_DEPOT_FILE,
}
_NODE_ZOOM_OVERRIDE = {
    'VER': _REINF_DEPOT_ZOOM,
    'LJQ': _REINF_DEPOT_ZOOM,
}

# Road style
_EDGE_STYLE = {
    'primary': ((30, 30, 30),  4.0, False),
    'secondary': ((85, 85, 85),  2.6, True),
    'tertiary': ((85, 85, 85),  2.0, True),
}

_LABEL_BLUE = (65, 105, 225)
_LABEL_RED  = (196, 30, 58)



class PygameRenderer:
    """
    Pygame-based live renderer. API-compatible with Env.renderer.MapRenderer.

    Parameters
    env : PeninsularWarEnv
        Already reset.
    window : (int, int)
        Initial window size in pixels. Default (1500, 1100).
    base_width : int
        Internal render resolution width for the map "world" surface. The map
        is drawn once at this resolution and zoom/pan scale a region of it onto
        the window. Higher = sharper when zoomed, slower. Default 1600.
    delay : float
        Seconds the window stays interactive per turn before update() returns
        (the playback pace). Default 1.5.
    fps : int
        Redraw rate of the interactive loop. Default 30.
    record : str | None
        If given, path to an .mp4 to write. Falls back to PNG frames if
        imageio/ffmpeg is unavailable.
    record_every_frame : bool
        If True and recording, capture every interactive frame (smooth video,
        large files). If False (default), capture one frame per turn.
    """

    def __init__(
        self,
        env: 'PeninsularWarEnv',
        window: tuple[int, int] = (1500, 1100),
        base_width: int = 1600,
        delay: float = 1.5,
        fps: int = 30,
        record: Optional[str] = None,
        record_every_frame: bool = False,
    ):
        self.env = env
        self.delay = float(delay)
        self.fps = int(fps)
        self.closed = False
        self.paused = False
        self._step_once = False
        self._record_every_frame = bool(record_every_frame)

        root = Path(__file__).parent.parent
        self._map_dir = root / 'Map'

        # Static node / edge metadata
        nodes_df = pd.read_csv(self._map_dir / 'nodes.csv')
        edges_df = pd.read_csv(self._map_dir / 'edges.csv')

        self._ntype = (
            nodes_df.set_index('node_id')['node_type'].to_dict()
            if 'node_type' in nodes_df.columns
            else {nid: 'town' for nid in env.node_ids}
        )

        # Pygame init
        pygame.init()
        pygame.display.set_caption('Peninsular War - simulation')
        self.win_w, self.win_h = window
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self._font_sm = pygame.font.SysFont('dejavusans', 11, bold=True)
        self._font_md = pygame.font.SysFont('dejavusans', 14, bold=True)
        self._font_hud = pygame.font.SysFont('dejavusansmono', 16, bold=True)
        self._font_help = pygame.font.SysFont('dejavusans', 13)

        # Background image
        map_path = self._map_dir / 'Iber_Pen_Topo_Map.jpg'
        if map_path.exists():
            bg = pygame.image.load(str(map_path)).convert()
            self._native_w, self._native_h = bg.get_size()
        else:
            self._native_w, self._native_h = 2499, 1878
            bg = pygame.Surface((self._native_w, self._native_h))
            bg.fill((40, 44, 52))

        self.base_w = int(base_width)
        self.base_h = int(round(self.base_w * self._native_h / self._native_w))
        self._scale = self.base_w / self._native_w   # Native px -> base px
        self._bg_base = pygame.transform.smoothscale(bg, (self.base_w, self.base_h))

        # Pre-compute base-surface positions for every node
        self._pos: dict[str, tuple[float, float]] = {}
        for nid in env.node_ids:
            nx, ny = _to_native_px(env._lat[nid], env._lon[nid])
            self._pos[nid] = (nx * self._scale, ny * self._scale)

        # Pre-render the static layer (background + roads) once
        self._static = self._bg_base.copy()
        self._draw_edges(self._static, edges_df)

        # Cache scaled node symbols and flags at base resolution
        self._sym_cache: dict[str, pygame.Surface] = {}
        self._sym_tint_cache: dict[tuple, pygame.Surface] = {}   # (key, owner) -> surf
        self._flag_cache: dict[str, pygame.Surface] = {}
        self._node_sym_key: dict[str, str] = {}
        self._preload_symbols(env)
        self._preload_flags()

        # Camera (fit map to window)
        self.zoom = 1.0
        self.cam_x = 0.0   # top-left of visible region, in base-surface px
        self.cam_y = 0.0
        self._fit_view()

        # Recording
        self._writer = None
        self._png_dir: Optional[Path] = None
        self._png_idx = 0
        if record:
            self._setup_recording(record)

        # Dragging state
        self._dragging = False
        self._drag_anchor = (0, 0)
        self._cam_anchor = (0.0, 0.0)

        self._battle_log: List[dict] = []

        # First paint
        self._render_frame()
        pygame.display.flip()


    def _load_png(self, filename: str) -> pygame.Surface:
        return pygame.image.load(str(self._map_dir / filename)).convert_alpha()

    @staticmethod
    def _tint_surface(surf: pygame.Surface, rgb) -> pygame.Surface:
        """
        Return a copy of `surf` recoloured to `rgb`, preserving its shading
        and alpha. Brightest opaque pixel maps to full `rgb`; darker pixels scale
        down — same brightness-preserving tint used in Map/visualise_graph.py.
        """
        out = surf.copy()
        rgb_view = pygame.surfarray.pixels3d(out)      # (w, h, 3) live view
        a_view = pygame.surfarray.pixels_alpha(out)    # (w, h) live view
        lum = rgb_view.astype(np.float32).mean(axis=2)
        mask = a_view > 0
        mx = lum[mask].max() if mask.any() else 255.0
        v = (lum / (mx or 1.0)).clip(0, 1)
        for c in range(3):
            rgb_view[..., c] = (rgb[c] * v).astype(np.uint8)
        del rgb_view, a_view                           # release surface locks
        return out

    def _tinted_symbol(self, key: str, owner: int):
        """
        Faction-tinted variant of a cached base symbol (lazily built).
        """
        base = self._sym_cache.get(key)
        if base is None or owner not in _NODE_FC:
            return base
        ck = (key, owner)
        if ck not in self._sym_tint_cache:
            self._sym_tint_cache[ck] = self._tint_surface(base, _NODE_FC[owner])
        return self._sym_tint_cache[ck]

    def _preload_symbols(self, env):
        for nid in env.node_ids:
            ntype = self._ntype.get(nid, 'town')
            fname = _NODE_IMAGE_OVERRIDE.get(nid, _NODE_IMAGE_FILE.get(ntype, _NODE_IMAGE_FILE['town']))
            zoom  = _NODE_ZOOM_OVERRIDE.get(nid, _NODE_ZOOM.get(ntype, _NODE_ZOOM['town']))
            key = f'{fname}@{zoom}'
            self._node_sym_key[nid] = key
            if key in self._sym_cache:
                continue
            try:
                img = self._load_png(fname)
            except Exception:
                surf = pygame.Surface((8, 8), pygame.SRCALPHA)
                pygame.draw.circle(surf, (250, 250, 250), (4, 4), 4)
                self._sym_cache[key] = surf
                continue
            tw = max(6, int(img.get_width() * zoom * (self.base_w / self._native_w) * 5.2))
            th = max(6, int(img.get_height() * zoom * (self.base_w / self._native_w) * 5.2))
            self._sym_cache[key] = pygame.transform.smoothscale(img, (tw, th))

    def _preload_flags(self):
        files = {
            'FR': 'Flags/French_Flag.png',
            'UK': 'Flags/British_Flag.png',
            'ES': 'Flags/Spanish_Flag.png',
            'PT': 'Flags/Portuguese_Flag.png',
        }
        # Visible faction colours for the fallback block (so a failed load still
        # shows *something* recognisable rather than a grey box / nothing).
        fallback_rgb = {'FR': (0, 60, 160), 'UK': (180, 30, 40),
                        'ES': (200, 30, 40), 'PT': (20, 110, 70)}
        # Cache each flag at a generous reference width; per-army flags are
        # scaled DOWN from this at draw time so flag size can encode army size.
        self._flag_ref_w = max(48, int(0.045 * self.base_w))
        ref_w = self._flag_ref_w
        report = []
        for key, fn in files.items():
            try:
                img = self._load_png(fn)
                h = max(6, int(ref_w * img.get_height() / img.get_width()))
                self._flag_cache[key] = pygame.transform.smoothscale(img, (ref_w, h))
                report.append(f'{key}(ok)')
            except Exception as e:
                surf = pygame.Surface((ref_w, int(ref_w * 0.62)), pygame.SRCALPHA)
                surf.fill(fallback_rgb.get(key, (200, 200, 200)))
                self._flag_cache[key] = surf
                report.append(f'{key}(FALLBACK: {type(e).__name__})')
        print('[pygame_renderer] flags ->', ' '.join(report), flush=True)

    # Static edge layer

    def _draw_edges(self, surf: pygame.Surface, edges_df: pd.DataFrame):
        for _, row in edges_df.iterrows():
            n1, n2 = row['node1'], row['node2']
            if n1 not in self._pos or n2 not in self._pos:
                continue
            rt = row.get('road_type', 'secondary')
            col, w, dashed = _EDGE_STYLE.get(rt, _EDGE_STYLE['secondary'])
            p1 = self._pos[n1]
            p2 = self._pos[n2]
            width = max(1, int(round(w * self._scale)))
            if dashed:
                self._draw_dashed(surf, col, p1, p2, width)
            else:
                pygame.draw.line(surf, col, p1, p2, width)

    @staticmethod
    def _draw_dashed(surf, col, p1, p2, width, dash=10, gap=7):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        dist = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / dist, dy / dist
        n = int(dist // (dash + gap)) + 1
        for i in range(n):
            s = i * (dash + gap)
            e = min(s + dash, dist)
            sx, sy = x1 + ux * s, y1 + uy * s
            ex, ey = x1 + ux * e, y1 + uy * e
            pygame.draw.line(surf, col, (sx, sy), (ex, ey), width)

    # Camera helpers

    def _fit_view(self):
        """Reset zoom/pan so the whole map fits the window."""
        self.zoom = min(self.win_w / self.base_w, self.win_h / self.base_h)
        view_w = self.win_w / self.zoom
        view_h = self.win_h / self.zoom
        self.cam_x = (self.base_w - view_w) / 2
        self.cam_y = (self.base_h - view_h) / 2

    def _clamp_cam(self):
        view_w = self.win_w / self.zoom
        view_h = self.win_h / self.zoom
        if view_w >= self.base_w:
            self.cam_x = (self.base_w - view_w) / 2
        else:
            self.cam_x = min(max(self.cam_x, 0), self.base_w - view_w)
        if view_h >= self.base_h:
            self.cam_y = (self.base_h - view_h) / 2
        else:
            self.cam_y = min(max(self.cam_y, 0), self.base_h - view_h)

    def _zoom_at(self, mx, my, factor):
        wx = self.cam_x + mx / self.zoom
        wy = self.cam_y + my / self.zoom
        min_zoom = min(self.win_w / self.base_w, self.win_h / self.base_h)
        self.zoom = float(np.clip(self.zoom * factor, min_zoom, 8.0))
        self.cam_x = wx - mx / self.zoom
        self.cam_y = wy - my / self.zoom
        self._clamp_cam()


    def update(self, battle_log: Optional[List[dict]] = None) -> bool:
        """
        Show the current env state and stay interactive for `delay` seconds
        (or until the user steps/quits). Returns False if the window was closed.
        """
        if self.closed:
            return False
        self._battle_log = battle_log or []

        captured_this_turn = False
        start = pygame.time.get_ticks()
        while not self.closed:
            self._handle_events()
            if self.closed:
                break
            self._render_frame()
            pygame.display.flip()

            if self._record_every_frame:
                self._capture()
            elif not captured_this_turn:
                self._capture()
                captured_this_turn = True

            self.clock.tick(self.fps)

            if self.paused:
                if self._step_once:
                    self._step_once = False
                    break
                continue   # stay paused, keep pumping events
            if pygame.time.get_ticks() - start >= self.delay * 1000:
                break
        return not self.closed

    def close(self):
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        self.closed = True
        try:
            pygame.quit()
        except Exception:
            pass

    def keep_open(self):
        """
        Block until the user closes the window (call after the game ends).
        """
        while not self.closed:
            self._handle_events()
            if self.closed:        # window/ESC closed during event handling:
                break              # pygame is already torn down, don't render
            self._render_frame()
            pygame.display.flip()
            self.clock.tick(self.fps)

    # Event handling

    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close()
            elif ev.type == pygame.VIDEORESIZE:
                self.win_w, self.win_h = ev.w, ev.h
                self.screen = pygame.display.set_mode(
                    (self.win_w, self.win_h), pygame.RESIZABLE)
                self._clamp_cam()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.close()
                elif ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif ev.key in (pygame.K_RIGHT, pygame.K_PERIOD):
                    if self.paused:
                        self._step_once = True
                elif ev.key == pygame.K_r:
                    self._fit_view()
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    self._dragging = True
                    self._drag_anchor = ev.pos
                    self._cam_anchor = (self.cam_x, self.cam_y)
            elif ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                self._zoom_at(mx, my, 1.15 ** ev.y)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    self._dragging = False
            elif ev.type == pygame.MOUSEMOTION and self._dragging:
                dx = (ev.pos[0] - self._drag_anchor[0]) / self.zoom
                dy = (ev.pos[1] - self._drag_anchor[1]) / self.zoom
                self.cam_x = self._cam_anchor[0] - dx
                self.cam_y = self._cam_anchor[1] - dy
                self._clamp_cam()

    # Rendering

    def _render_frame(self):
        # Compose the world (static + dynamic) onto a copy of the static layer
        world = self._static.copy()
        self._draw_nodes(world)
        self._draw_flags(world)
        if self._battle_log:
            self._draw_battles(world)

        # Apply camera: take the visible slice of the world that actually lies
        # inside the base surface, scale just that slice to the window, and
        # letterbox the rest. This stays valid when the fitted view is larger
        # than the map (differing aspect ratios) and is cheap when zoomed in.
        vw = self.win_w / self.zoom
        vh = self.win_h / self.zoom
        sx0 = max(0, int(self.cam_x))
        sy0 = max(0, int(self.cam_y))
        sx1 = min(self.base_w, int(self.cam_x + vw) + 1)
        sy1 = min(self.base_h, int(self.cam_y + vh) + 1)
        self.screen.fill((0, 0, 0))
        sw, sh = sx1 - sx0, sy1 - sy0
        if sw > 0 and sh > 0:
            region = world.subsurface(pygame.Rect(sx0, sy0, sw, sh))
            dest_w = max(1, int(round(sw * self.zoom)))
            dest_h = max(1, int(round(sh * self.zoom)))
            dest_x = int(round((sx0 - self.cam_x) * self.zoom))
            dest_y = int(round((sy0 - self.cam_y) * self.zoom))
            scaled = pygame.transform.smoothscale(region, (dest_w, dest_h))
            self.screen.blit(scaled, (dest_x, dest_y))

        # HUD overlays (screen space, not zoomed)
        self._draw_scoreboard()
        self._draw_help()

    def _draw_nodes(self, surf):
        env = self.env
        for i, nid in enumerate(env.node_ids):
            owner = int(env.owner[i])
            ntype = self._ntype.get(nid, 'town')
            x, y = self._pos[nid]
            # symbol, tinted by faction owner (France blue / Allies red / Neutral grey)
            sym = self._tinted_symbol(self._node_sym_key[nid], owner)
            sym_h = sym.get_height() if sym is not None else 0
            if sym is not None:
                surf.blit(sym, (x - sym.get_width() / 2, y - sym_h / 2))
            # node id label — only for the more important places (city and up);
            # towns and intersections are left unlabelled to reduce clutter.
            if ntype in ('capital', 'regional_capital', 'major_city', 'city'):
                self._blit_text(surf, self._font_sm, nid, (255, 255, 255),
                                (x, y + sym_h / 2 + 1), centre=True, bg=(0, 0, 0, 140))

    def _draw_flags(self, surf):
        env = self.env
        allied_flag = {
            SUBFACTION_BRITISH:    'UK',
            SUBFACTION_SPANISH:    'ES',
            SUBFACTION_PORTUGUESE: 'PT',
        }
        for i, nid in enumerate(env.node_ids):
            x, y = self._pos[nid]
            f_men  = int(env.france_infantry[i] + env.france_cavalry[i])
            f_arty = int(env.france_artillery[i])
            a_men  = int(env.allies_infantry[i] + env.allies_cavalry[i])
            a_arty = int(env.allies_artillery[i])

            fr_present = f_men > 0 or f_arty > 0
            al_present = a_men > 0 or a_arty > 0

            # When both factions occupy a node, offset the two flags sideways so
            # they don't overlap; otherwise centre the flag on the node.
            offset = 0.45 * self._flag_ref_w
            if fr_present:
                fx = x - (offset if al_present else 0)
                self._blit_flag(surf, 'FR', fx, y, f_men, _LABEL_BLUE)
            if al_present:
                sf = int(env.sub_faction[i])
                fx = x + (offset if fr_present else 0)
                self._blit_flag(surf, allied_flag.get(sf, 'UK'), fx, y, a_men, _LABEL_RED)

    def _flag_width_for(self, men):
        # Fixed flag width for all armies;
        # the army size is shown as the number to the right of the flag.
        return min(self._flag_ref_w, max(22, int(0.020 * self.base_w)))

    def _blit_flag(self, surf, key, x, y, men, label_bg):
        ref = self._flag_cache.get(key)
        if ref is None:
            return
        tw = self._flag_width_for(men)
        th = max(5, int(tw * ref.get_height() / ref.get_width()))
        flag = pygame.transform.smoothscale(ref, (tw, th))
        fx = x - tw / 2
        fy = y - th - max(3, int(4 * self._scale))    # sit just above the node
        pygame.draw.rect(surf, (20, 20, 20),
                         pygame.Rect(fx - 1, fy - 1, tw + 2, th + 2))
        surf.blit(flag, (fx, fy))
        # army size (men) to the RIGHT of the flag, vertically centred on it
        label = f'{men // 1000}k' if men >= 1000 else str(int(men))
        lx = fx + tw + max(2, int(3 * self._scale))
        ly = fy + th / 2 - self._font_sm.get_height() / 2
        self._blit_text(surf, self._font_sm, label, (255, 255, 255),
                        (lx, ly), centre=False, bg=(*label_bg, 220))

    def _draw_battles(self, surf):
        for b in self._battle_log:
            nid = b.get('node')
            if nid not in self._pos:
                continue
            x, y = self._pos[nid]
            r = max(8, int(14 * self._scale))
            col = (255, 40, 40)
            w = max(2, int(3 * self._scale))
            pygame.draw.line(surf, col, (x - r, y - r), (x + r, y + r), w)
            pygame.draw.line(surf, col, (x - r, y + r), (x + r, y - r), w)

    # Text helper

    def _blit_text(self, surf, font, text, colour, pos, centre=False,
                   bg=None, anchor_bottom=False):
        img = font.render(text, True, colour)
        w, h = img.get_size()
        x, y = pos
        if centre:
            x -= w / 2
        if anchor_bottom:
            y -= h
        if bg is not None:
            pad = 2
            box = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
            box.fill(bg)
            surf.blit(box, (x - pad, y - pad))
        surf.blit(img, (x, y))

    # HUD

    def _draw_scoreboard(self):
        env = self.env
        f_t = int(np.sum(env.france_infantry + env.france_cavalry))
        a_t = int(np.sum(env.allies_infantry + env.allies_cavalry))
        f_g = int(np.sum(env.france_artillery))
        a_g = int(np.sum(env.allies_artillery))
        f_n = int(np.sum(env.owner == FRANCE))
        a_n = int(np.sum(env.owner == ALLIES))
        t = env.turn

        lines = [
            (f'Turn {t:>3} / {MAX_TURNS}   ({t / 52:.1f} yrs)', (235, 235, 235)),
            (f'France  {f_t:>8,} men  {f_g:>4} g  {f_n:>3} nodes', _LABEL_BLUE),
            (f'Allies  {a_t:>8,} men  {a_g:>4} g  {a_n:>3} nodes', _LABEL_RED),
        ]
        imgs = [self._font_hud.render(s, True, c) for s, c in lines]
        w = max(i.get_width() for i in imgs) + 20
        h = sum(i.get_height() for i in imgs) + 16
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((17, 17, 17, 210))
        pygame.draw.rect(panel, (90, 90, 90), panel.get_rect(), 1)
        yy = 8
        for img in imgs:
            panel.blit(img, (10, yy))
            yy += img.get_height()
        self.screen.blit(panel, (self.win_w - w - 12, 12))

    def _draw_help(self):
        status = 'PAUSED' if self.paused else 'playing'
        txt = (f'[{status}]  SPACE pause | -> step | drag pan | '
               f'wheel zoom | R reset | ESC quit')
        img = self._font_help.render(txt, True, (230, 230, 230))
        w, h = img.get_size()
        panel = pygame.Surface((w + 16, h + 10), pygame.SRCALPHA)
        panel.fill((17, 17, 17, 190))
        panel.blit(img, (8, 5))
        self.screen.blit(panel, (12, self.win_h - h - 18))

    # Recording

    def _setup_recording(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio  # noqa
            fps = self.fps if self._record_every_frame else max(1, int(1 / max(self.delay, 1e-3)))
            self._writer = imageio.get_writer(str(p), fps=fps, macro_block_size=None)
            print(f'[pygame_renderer] recording MP4 -> {p}')
        except Exception as e:
            self._writer = None
            self._png_dir = p.with_suffix('')
            self._png_dir.mkdir(parents=True, exist_ok=True)
            print(f'[pygame_renderer] imageio/ffmpeg unavailable ({e}); '
                  f'writing PNG frames -> {self._png_dir}')

    def snapshot(self, path, add_to_video=False):
        """
        Render the current env state and save it as a PNG.
        """
        self._render_frame()
        pygame.display.flip()
        try:
            pygame.image.save(self.screen, str(path))
            print(f'[pygame_renderer] saved snapshot -> {path}')
        except Exception as e:
            print(f'[pygame_renderer] snapshot failed: {e}')
        if add_to_video:
            self._capture()

    def _capture(self):
        if self._writer is None and self._png_dir is None:
            return
        # surfarray is (w, h, 3); transpose to (h, w, 3) for image libs
        arr = pygame.surfarray.array3d(self.screen)
        arr = np.transpose(arr, (1, 0, 2))
        if self._writer is not None:
            self._writer.append_data(arr)
        else:
            from PIL import Image
            Image.fromarray(arr).save(self._png_dir / f'frame_{self._png_idx:05d}.png')
            self._png_idx += 1
