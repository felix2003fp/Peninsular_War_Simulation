# Makes Map/ an importable package so its shared helpers (e.g. map_projection)
# can be imported as `from Map.map_projection import ...` from anywhere in the
# project. The data files (nodes.csv, edges.csv, images) are still read by path.
