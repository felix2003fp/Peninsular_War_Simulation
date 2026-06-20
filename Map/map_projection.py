"""
map_projection.py
-----------------
Single source of truth for (a) the lat/lon -> topo-map-pixel projection and
(b) the node-type -> symbol-image filename map. Shared by Env/renderer.py,
Env/pygame_renderer.py and Map/visualise_graph.py so the calibration and the
symbol assignment live in exactly one place.

This module is side-effect free (it only defines data and pure functions), so
it is safe to import from anywhere.

Calibration anchors were measured directly on Iber_Pen_Topo_Map.jpg
(2499 x 1878 px); the lat/lon values are the authoritative ones from nodes.csv.
If you re-measure a point, edit it here only.
"""

import numpy as np

# (lat, lon, pixel_x, pixel_y)
CALIBRATION_POINTS = [
    (41.3888,  2.1590, 2056,  593),   # Barcelona
    (42.3588,  1.4614, 1935,  365),   # Urgell
    (38.7251, -9.1498,  134, 1153),   # Lisbon
    (36.5267, -6.2891,  616, 1668),   # Cádiz
    (43.4659, -3.8049, 1072,  136),   # Santander
    (41.9831,  2.8249, 2159,  425),   # Girona
    (43.3713, -8.3960,  325,  132),   # La Coruña
    (36.8381, -2.4597, 1301, 1608),   # Almería
    (37.0187, -7.9272,  328, 1550),   # Faro
    (40.4165, -3.7026, 1089,  813),   # Madrid
    (39.4739, -0.3797, 1659, 1018),   # Valencia
    (38.3452, -0.4815, 1645, 1266),   # Alicante
    (38.8779, -6.9706,  514, 1141),   # Badajoz
    (43.2627, -2.9253, 1208,  174),   # Bilbao
    (41.6561, -0.8773, 1562,  539),   # Zaragoza
    (36.7202, -4.4203,  949, 1633),   # Málaga
    (41.1485, -8.6110,  253,  618),   # Oporto
]


def _fit(points):
    """Quadratic least-squares fit  lat/lon -> pixel:
        p = c0 + c1·lon + c2·lat + c3·lon·lat + c4·lon² + c5·lat²
    The lon·lat term models meridian convergence (a plain affine cannot, which
    pushed mid-map cities like Valencia ~25 px inland). RMS ≈ 6.6 px.
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
    """Map (lat, lon) to a pixel (x, y) in the native topo-map image frame."""
    v = np.array([1.0, lon, lat, lon * lat, lon * lon, lat * lat])
    return float(v @ _CX), float(v @ _CY)


# node_type -> symbol-image filename (relative to Map/)
NODE_SYMBOL_FILE = {
    'capital':          'Node_Symbols/Capital_Symbol.png',
    'regional_capital': 'Node_Symbols/Regional_Capital_Symbol.png',
    'major_city':       'Node_Symbols/Major_City_Symbol.png',
    'city':             'Node_Symbols/City_Symbol.png',
    'town':             'Node_Symbols/Town_Symbol.png',
    'intersection':     'Node_Symbols/Intersection_Symbol.png',
}
REINF_DEPOT_FILE = 'Node_Symbols/Reinf_Depot_Symbol.png'
