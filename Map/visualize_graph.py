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

# Configuration

MAP_DIR     = Path(__file__).parent
NODES_FILE  = "Map/nodes.csv"
EDGES_FILE  = "Map/edges.csv"
MAP_FILE    = "Map/Iber_Pen_Topo_Map.jpg"
OUTPUT_FILE = "Map/Charted_Map.jpg"

# Calibration points: (lat, lon, pixel_x, pixel_y)
# Measured directly from Iber_Pen_Topo_Map.jpg.

# The lat/lon to pixel projection and node-symbol map are shared with the
# renderers, defined once in Map/map_projection.py.
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from Map.map_projection import latlon_to_pixel

# Node symbol images

# node_type to symbol filename
from Map.map_projection import (
    NODE_SYMBOL_FILE as NODE_IMAGE_FILE,
    REINF_DEPOT_FILE,
)

NODE_IMAGE_OVERRIDE = {
    'VER': REINF_DEPOT_FILE,   # Verdun
    'LJQ': REINF_DEPOT_FILE,   # La Jonquera
}

# node_type to on-map zoom factor for its symbol image
NODE_ZOOM = {
    'capital':          0.12,
    'regional_capital': 0.102,
    'major_city':       0.085,
    'city':             0.067,
    'town':             0.056,
    'intersection':     0.045,
}
REINF_DEPOT_ZOOM = 0.085

# node_id -> zoom override
NODE_ZOOM_OVERRIDE = {
    'VER': REINF_DEPOT_ZOOM,
    'LJQ': REINF_DEPOT_ZOOM,
}

_IMAGE_CACHE = {}


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



# Edge style table

# road style depending on road type
EDGE_STYLE = {
    'primary':   ('black', 2.0, 0.90, '-'),
    'secondary': ('#555555', 1.2, 0.75, '--'),
    'tertiary':  ('#555555', 1.0, 0.65, ':'),
}

# node_types whose labels are drawn on the map
LABEL_TYPES = {'capital', 'regional_capital', 'major_city'}




def main():
    nodes = pd.read_csv(NODES_FILE)
    edges = pd.read_csv(EDGES_FILE)

    # Fallback if the CSVs haven't been enriched yet
    if 'node_type' not in nodes.columns:
        nodes['node_type'] = 'town'
        print("nodes.csv has no 'node_type' column -- run enrich_map.py first.")
    if 'road_type' not in edges.columns:
        edges['road_type'] = 'secondary'
        print("edges.csv has no 'road_type' column -- run enrich_map.py first.")

    img = mpimg.imread(MAP_FILE)
    img_height, img_width = img.shape[:2]
    print(f"Image: {img_width} x {img_height} px")

    # Build node lookup: node_id to (px, py, full_name, node_type)
    node_pixels = {}
    for _, row in nodes.iterrows():
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            continue
        px, py = latlon_to_pixel(row['latitude'], row['longitude'])
        ntype  = row.get('node_type', 'town')
        node_pixels[row['node_id']] = (px, py, row['full_name'], ntype)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 16 * img_height / img_width))
    ax.imshow(img, extent=[0, img_width, img_height, 0])
    ax.axis('off')
    fig.patch.set_facecolor('black')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Edges
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

    # Nodes
    for node_id, (px, py, full_name, ntype) in node_pixels.items():
        img_arr, zoom = node_symbol(node_id, ntype)
        imbox = OffsetImage(img_arr, zoom=zoom)
        ab = AnnotationBbox(imbox, (px, py), frameon=False, zorder=3,
                             box_alignment=(0.5, 0.5))
        ax.add_artist(ab)


    # Legend
    node_legend_order = [
        ('capital',          'Capital'),
        ('regional_capital', 'Regional capital'),
        ('major_city',       'Major city'),
        ('city',             'City'),
        ('town',             'Town'),
        ('intersection',     'Intersection'),
        (None,               'Reinf. depot'),
    ]
    road_legend_order = [
        ('primary',   'Primary road'),
        ('secondary', 'Secondary road'),
        ('tertiary',  'Tertiary road'),
    ]

    row_h   = 0.022
    pad     = 0.007
    title_h = 0.020
    panel_w = 0.120
    n_rows  = len(node_legend_order) + len(road_legend_order)
    panel_h = n_rows * row_h + 2 * title_h + 2 * pad
    panel_x0 = 1.0 - panel_w - 0.010          
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

    # Save
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
