from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAP_DIR     = Path(__file__).parent
NODES_FILE  = "Map/nodes.csv"
EDGES_FILE  = "Map/edges.csv"
MAP_FILE    = "Map/Iber_Pen_Topo_Map.jpg"
OUTPUT_FILE = "Map/Charted_Map.jpg"

# ---------------------------------------------------------------------------
# Calibration points: (lat, lon, pixel_x, pixel_y)
# Measured directly from Iber_Pen_Topo_Map.jpg.
# ---------------------------------------------------------------------------

CALIBRATION_POINTS = [
    #  lat,      lon,       px,    py     (lat/lon are authoritative from nodes.csv)
    (  41.3888,  2.1590,  2056,   593),   # Barcelona
    (  42.3588,  1.4614,  1935,   365),   # Urgell
    (  38.7251, -9.1498,   134,  1153),   # Lisbon
    (  36.5267, -6.2891,   616,  1668),   # Cadiz
    (  43.4659, -3.8049,  1072,   136),   # Santander
    (  41.9831,  2.8249,  2159,   425),   # Girona
    (  43.3713, -8.3960,   325,   132),   # La Coruna
    (  36.8381, -2.4597,  1301,  1608),   # Almeria
    (  37.0187, -7.9272,   328,  1550),   # Faro
    (  40.4165, -3.7026,  1089,   813),   # Madridxº
    (  39.4739, -0.3797,  1659,  1018),   # Valencia
    (  38.3452, -0.4815,  1645,  1266),   # Alicante
    (  38.8779, -6.9706,   514,  1141),   # Badajoz
    (  43.2627, -2.9253,  1208,   174),   # Bilbao
    (  41.6561, -0.8773,  1562,   539),   # Zaragoza
    (  36.7202, -4.4203,   949,  1633),   # Malaga
    (  41.1485, -8.6110,   253,   618),   # Oporto
]

# ---------------------------------------------------------------------------
# Coordinate conversion (fitted once at startup)
# ---------------------------------------------------------------------------

def _fit(points):
    """Quadratic least-squares fit lat/lon -> pixel:
        p = c0 + c1*lon + c2*lat + c3*lon*lat + c4*lon^2 + c5*lat^2
    The lon*lat term captures meridian convergence, which a plain affine cannot
    (the old affine pushed mid-map cities like Valencia ~25 px inland). RMS ~ 6.6 px.
    """
    lat = np.array([p[0] for p in points], float)
    lon = np.array([p[1] for p in points], float)
    pxs = np.array([p[2] for p in points], float)
    pys = np.array([p[3] for p in points], float)
    A = np.column_stack([np.ones_like(lon), lon, lat, lon * lat, lon * lon, lat * lat])
    cx, *_ = np.linalg.lstsq(A, pxs, rcond=None)
    cy, *_ = np.linalg.lstsq(A, pys, rcond=None)
    return cx, cy

_CX, _CY = _fit(CALIBRATION_POINTS)

def latlon_to_pixel(lat, lon):
    v = np.array([1.0, lon, lat, lon * lat, lon * lon, lat * lat])
    return float(v @ _CX), float(v @ _CY)

# ---------------------------------------------------------------------------
# Node symbol images
# ---------------------------------------------------------------------------

# node_type -> symbol filename
NODE_IMAGE_FILE = {
    'capital':          'Node_Symbols/Capital_Symbol.png',
    'regional_capital': 'Node_Symbols/Regional_Capital_Symbol.png',
    'major_city':       'Node_Symbols/Major_City_Symbol.png',
    'city':             'Node_Symbols/City_Symbol.png',
    'town':             'Node_Symbols/Town_Symbol.png',
    'intersection':     'Node_Symbols/Intersection_Symbol.png',
}
REINF_DEPOT_FILE = 'Node_Symbols/Reinf_Depot_Symbol.png'

NODE_IMAGE_OVERRIDE = {
    'VER': REINF_DEPOT_FILE,   # Verdun  -> reinforcement depot
    'LJQ': REINF_DEPOT_FILE,   # La Jonquera -> reinforcement depot
}

# node_type -> on-map zoom factor for its symbol image
NODE_ZOOM = {
    'capital':          0.12,
    'regional_capital': 0.102,
    'major_city':       0.085,
    'city':             0.067,
    'town':             0.056,
    'intersection':     0.045,
}
REINF_DEPOT_ZOOM = 0.096

# node_id -> zoom override (paired with NODE_IMAGE_OVERRIDE)
NODE_ZOOM_OVERRIDE = {
    'VER': REINF_DEPOT_ZOOM,
    'LJQ': REINF_DEPOT_ZOOM,
}

_IMAGE_CACHE = {}
_TINT_CACHE = {}


def _load_image(filename):
    if filename not in _IMAGE_CACHE:
        path = MAP_DIR / filename
        _IMAGE_CACHE[filename] = np.array(Image.open(path).convert('RGBA'))
    return _IMAGE_CACHE[filename]

def node_symbol(node_id, ntype):
    """Return (image array, zoom factor) for the given node."""
    filename = NODE_IMAGE_OVERRIDE.get(node_id, NODE_IMAGE_FILE.get(ntype, NODE_IMAGE_FILE['town']))
    zoom     = NODE_ZOOM_OVERRIDE.get(node_id, NODE_ZOOM.get(ntype, NODE_ZOOM['town']))
    return _load_image(filename), zoom

# ---------------------------------------------------------------------------
# Edge style table
# ---------------------------------------------------------------------------

# road_type -> (color, linewidth, alpha, linestyle)
EDGE_STYLE = {
    'primary':   ('black', 2.0, 0.90, '-'),
    'secondary': ('#555555', 1.2, 0.75, '--'),
    'tertiary':  ('#555555', 1.0, 0.65, ':'),
}

# node_types whose labels are drawn on the map
LABEL_TYPES = {'capital', 'regional_capital', 'major_city'}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    nodes = pd.read_csv(NODES_FILE)
    edges = pd.read_csv(EDGES_FILE)

    # -- Fallback if the CSVs haven't been enriched yet ------------
    if 'node_type' not in nodes.columns:
        nodes['node_type'] = 'town'
        print("nodes.csv has no 'node_type' column -- run enrich_map.py first.")
    if 'road_type' not in edges.columns:
        edges['road_type'] = 'secondary'
        print("edges.csv has no 'road_type' column -- run enrich_map.py first.")

    img = mpimg.imread(MAP_FILE)
    img_height, img_width = img.shape[:2]
    print(f"Image: {img_width} x {img_height} px")

    # Build node lookup: node_id -> (px, py, full_name, node_type)
    node_pixels = {}
    for _, row in nodes.iterrows():
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            continue
        px, py = latlon_to_pixel(row['latitude'], row['longitude'])
        ntype  = row.get('node_type', 'town')
        node_pixels[row['node_id']] = (px, py, row['full_name'], ntype)

    # -- Plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 16 * img_height / img_width))
    ax.imshow(img, extent=[0, img_width, img_height, 0])
    ax.axis('off')
    fig.patch.set_facecolor('black')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # -- Edges (drawn first so nodes sit on top) -------------------------------
    skipped_edges = []
    for _, row in edges.iterrows():
        n1, n2 = row['node1'], row['node2']
        if n1 not in node_pixels or n2 not in node_pixels:
            skipped_edges.append(f"{row['edge_id']}: {n1} -- {n2}")
            continue
        x1, y1, *_ = node_pixels[n1]
        x2, y2, *_ = node_pixels[n2]
        rtype = row.get('road_type', 'secondary')
        col, lw, alpha, ls = EDGE_STYLE.get(rtype, EDGE_STYLE['secondary'])
        ax.plot([x1, x2], [y1, y2],
                color=col, linewidth=lw, alpha=alpha, linestyle=ls, zorder=1,
                solid_capstyle='round')

    # -- Nodes (PNG symbols) ----------------------------------------------------
    for node_id, (px, py, full_name, ntype) in node_pixels.items():
        img_arr, zoom = node_symbol(node_id, ntype)
        imbox = OffsetImage(img_arr, zoom=zoom)
        ab = AnnotationBbox(imbox, (px, py), frameon=False, zorder=3,
                             box_alignment=(0.5, 0.5))
        ax.add_artist(ab)


    # -- Legend -------------------------------------------------------------------
    node_legend_order = [
        ('capital',          'Capital'),
        ('regional_capital', 'Regional capital'),
        ('major_city',       'Major city'),
        ('city',             'City'),
        ('town',             'Town'),
        ('intersection',     'Intersection'),
        (None,               'Reinf. depot'),   # uses REINF_DEPOT_FILE
    ]
    road_legend_order = [
        ('primary',   'Primary road'),
        ('secondary', 'Secondary road'),
        ('tertiary',  'Tertiary road'),
    ]

    # Compact geometry (axes fraction)
    row_h   = 0.022
    pad     = 0.007
    title_h = 0.020
    panel_w = 0.120
    n_rows  = len(node_legend_order) + len(road_legend_order)
    panel_h = n_rows * row_h + 2 * title_h + 2 * pad
    panel_x0 = 1.0 - panel_w - 0.010          # bottom-RIGHT
    panel_y0 = 0.010
    icon_x   = panel_x0 + 0.020
    text_x   = panel_x0 + 0.040
    fs       = 5.5                        

    ax.add_patch(mpatches.FancyBboxPatch(
        (panel_x0, panel_y0), panel_w, panel_h,
        transform=ax.transAxes, zorder=9, boxstyle="round,pad=0.004",
        facecolor='#111111', edgecolor='#555555', alpha=0.78))

    y = panel_y0 + panel_h - pad
    # Nodes section
    ax.text(panel_x0 + panel_w / 2, y, 'Nodes', transform=ax.transAxes, zorder=10,
            fontsize=fs + 1.5, color='white', ha='center', va='top', weight='bold')
    y -= title_h
    for ntype, label in node_legend_order:
        yc = y - row_h / 2
        filename = REINF_DEPOT_FILE if ntype is None else NODE_IMAGE_FILE[ntype]
        ab = AnnotationBbox(OffsetImage(_load_image(filename), zoom=0.026),
                            (icon_x, yc), xycoords='axes fraction',
                            frameon=False, zorder=10, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
        ax.text(text_x, yc, label, transform=ax.transAxes, zorder=10,
                fontsize=fs, color='white', va='center', ha='left')
        y -= row_h
    # Roads section
    ax.text(panel_x0 + panel_w / 2, y, 'Roads', transform=ax.transAxes, zorder=10,
            fontsize=fs + 1.5, color='white', ha='center', va='top', weight='bold')
    y -= title_h
    for rtype, label in road_legend_order:
        yc = y - row_h / 2
        col, lw, alpha, ls = EDGE_STYLE[rtype]
        ax.plot([icon_x - 0.013, icon_x + 0.013], [yc, yc], transform=ax.transAxes,
                color=col, linewidth=max(lw, 1.0), alpha=alpha, linestyle=ls,
                zorder=10, solid_capstyle='round')
        ax.text(text_x, yc, label, transform=ax.transAxes, zorder=10,
                fontsize=fs, color='white', va='center', ha='left')
        y -= row_h

    # -- Save -----------------------------------------------------------------------
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='black')
    png_out = OUTPUT_FILE.rsplit('.', 1)[0] + '.png'
    pdf_out = OUTPUT_FILE.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(png_out, dpi=300, bbox_inches='tight', facecolor='black')
    plt.savefig(pdf_out, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved -> {OUTPUT_FILE}")
    print(f"Saved -> {png_out}  (300 dpi, for LaTeX)")
    print(f"Saved -> {pdf_out}  (vector wrapper, for LaTeX)")

    if skipped_edges:
        print(f"\nSkipped {len(skipped_edges)} edge(s) with missing nodes:")
        for e in skipped_edges:
            print(f"  {e}")

if __name__ == "__main__":
    main()
