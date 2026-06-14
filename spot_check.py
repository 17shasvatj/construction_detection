"""
Dump confirmed-grading pixel lat/lons for Google Earth spot-check.
Uses verified coordinate conversion. Reads bbox fresh from metadata.

python spot_check_coords.py
"""
import numpy as np, json
from pathlib import Path

DATA_DIR = Path("wendell_data")
meta = json.load(open(DATA_DIR/"metadata.json"))
BBOX = meta["bbox"]
quarters = meta["quarters"]
H, W = meta["shape"][1], meta["shape"][2]
T = len(quarters)

dw = np.load(DATA_DIR/"dw_cube.npy")
label_cube = np.load(DATA_DIR/"label_cube.npy")

def px_to_lonlat(x, y):
    lon = BBOX[0] + (x + 0.5)/W * (BBOX[2]-BBOX[0])
    lat = BBOX[3] - (y + 0.5)/H * (BBOX[3]-BBOX[1])
    return lon, lat

confirmed = ((label_cube==1).any(0)) & ((label_cube==2).any(0))
ys, xs = np.where(confirmed)
print(f"Confirmed grading pixels: {len(ys)}")
print(f"bbox used: {BBOX}\n")

np.random.seed(7)
idx = np.random.choice(len(ys), 20, replace=False)
print(f"{'#':>3}  {'lat,lon (paste into Google Earth)':>34}  {'bare_q':>8}  {'built_q':>8}")
for i, j in enumerate(idx):
    y, x = ys[j], xs[j]
    lon, lat = px_to_lonlat(x, y)
    bare_q = next((quarters[t] for t in range(T) if dw[t,y,x]==7), "?")
    built_q = next((quarters[t] for t in range(T) if dw[t,y,x]==6 and
                    any(dw[s,y,x]==7 for s in range(t))), "?")
    print(f"{i+1:>3}  {lat:.5f}, {lon:.5f}              {bare_q:>8}  {built_q:>8}")

print("\nFor each: Google Earth should show vegetation before bare_q,")
print("graded/bare earth around bare_q, and a building by built_q.")
print("Centroid sanity: these should all be within Babcock Ranch town.")