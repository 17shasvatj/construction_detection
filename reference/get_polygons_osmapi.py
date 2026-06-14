"""
Building polygons via raw OSM /api/map, split into quadrants (node cap workaround)
==================================================================================
python get_polygons_osmapi.py
"""
import numpy as np
import json
import requests
import time
import xml.etree.ElementTree as ET
from pathlib import Path

BBOX = [-78.42, 35.76, -78.38, 35.795]
DATA_DIR = Path("wendell_data")
with open(DATA_DIR/"metadata.json") as f:
    meta = json.load(f)
H, W = meta["shape"][1], meta["shape"][2]
quarters = meta["quarters"]

def fetch_bbox(w, s, e, n):
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={w},{s},{e},{n}"
    r = requests.get(url, timeout=180,
                     headers={"User-Agent":"construction-detection-research"})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    nodes = {nd.get("id"):(float(nd.get("lon")),float(nd.get("lat")))
             for nd in root.findall("node")}
    polys = []
    for way in root.findall("way"):
        tags = {t.get("k"):t.get("v") for t in way.findall("tag")}
        if "building" in tags:
            coords=[nodes[nd.get("ref")] for nd in way.findall("nd")
                    if nd.get("ref") in nodes]
            if len(coords)>=3:
                polys.append(coords)
    return polys

# Split into NxN grid until each request succeeds
def fetch_recursive(w, s, e, n, depth=0):
    try:
        polys = fetch_bbox(w, s, e, n)
        print(f"  {'  '*depth}[{w:.3f},{s:.3f},{e:.3f},{n:.3f}] -> {len(polys)} buildings")
        return polys
    except requests.exceptions.HTTPError as ex:
        if depth > 3:
            print(f"  {'  '*depth}giving up on this cell: {ex}")
            return []
        # split into 4
        mw, mn = (w+e)/2, (s+n)/2
        print(f"  {'  '*depth}splitting [{w:.3f},{s:.3f},{e:.3f},{n:.3f}]...")
        out = []
        for (a,b,c,d) in [(w,s,mw,mn),(mw,s,e,mn),(w,mn,mw,n),(mw,mn,e,n)]:
            out += fetch_recursive(a,b,c,d,depth+1)
            time.sleep(2)
        return out

print("Fetching buildings via OSM map API (quadrant split)...")
all_polys = fetch_recursive(*BBOX)
print(f"\nTotal: {len(all_polys)} building polygons")
json.dump(all_polys, open(DATA_DIR/"buildings_polygons.json","w"))

# Rasterize
from matplotlib.path import Path as MplPath
def to_px(lon,lat):
    return ((lon-BBOX[0])/(BBOX[2]-BBOX[0])*W, (BBOX[3]-lat)/(BBOX[3]-BBOX[1])*H)
all_mask=np.zeros((H,W),np.uint8)
for coords in all_polys:
    pts=[to_px(lo,la) for lo,la in coords]
    cxs=[p[0] for p in pts]; cys=[p[1] for p in pts]
    x0,x1=max(int(min(cxs)),0),min(int(max(cxs))+1,W)
    y0,y1=max(int(min(cys)),0),min(int(max(cys))+1,H)
    if x0>=x1 or y0>=y1: continue
    sy,sx=np.mgrid[y0:y1,x0:x1]
    inside=MplPath(pts).contains_points(np.column_stack([sx.ravel(),sy.ravel()])).reshape(sy.shape)
    all_mask[y0:y1,x0:x1][inside]=1

#old=np.load(DATA_DIR/'building_mask_all.npy').sum()
#print(f"Rasterized -> {all_mask.sum()} px (was {old} with centers)")
print(f"Rasterized -> {all_mask.sum()} px")

dw_2021=np.load(DATA_DIR/"dw_cube.npy")[0]
veg=np.isin(dw_2021,[1,2,5])
new_mask=((all_mask>0)&veg).astype(np.uint8)
np.save(DATA_DIR/"building_mask_all.npy",all_mask)
np.save(DATA_DIR/"building_mask_new.npy",new_mask)
print(f"New (was veg): {new_mask.sum()} | Pre-existing: {(all_mask>0).sum()-new_mask.sum()}")

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def norm_rgb(d):
    a=d[[2,1,0]].copy(); o=np.zeros_like(a)
    for i in range(3):
        b=a[i]; v=b[~np.isnan(b)]
        if len(v): lo,hi=np.percentile(v,2),np.percentile(v,98); o[i]=np.clip((b-lo)/(hi-lo+1e-8),0,1)
    return np.nan_to_num(np.transpose(o,(1,2,0)))
latest=np.load(DATA_DIR/f"composite_{quarters[-1]}.npy")
fig,ax=plt.subplots(1,2,figsize=(16,9))
ax[0].imshow(norm_rgb(latest)); ax[0].axis('off'); ax[0].set_title("RGB")
ax[1].imshow(norm_rgb(latest))
ax[1].imshow(np.where(new_mask,1,np.nan),cmap='autumn',alpha=0.7)
ax[1].set_title(f"Polygon footprints new={new_mask.sum()}px"); ax[1].axis('off')
plt.tight_layout(); plt.savefig(DATA_DIR/"polygon_footprints.png",dpi=130,bbox_inches='tight')
print("Saved polygon_footprints.png")