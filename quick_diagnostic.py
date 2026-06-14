import numpy as np
from pathlib import Path
DATA_DIR = Path("wendell_data")
dw = np.load(DATA_DIR/"dw_cube.npy")
T = dw.shape[0]

was_veg = np.isin(dw[0], [1,2,5])
now_built = dw[-1] == 6
now_bare = dw[-1] == 7

constructed = was_veg & now_built
grading = was_veg & now_bare

# Confirmed trajectory: veg→bare→built
confirmed = np.zeros_like(was_veg)
for y in range(dw.shape[1]):
    for x in range(dw.shape[2]):
        if not (was_veg[y,x] and now_built[y,x]):
            continue
        if any(dw[t,y,x] == 7 for t in range(T)):
            confirmed[y,x] = True

print(f"was_veg in 2021: {was_veg.sum()} ({was_veg.mean()*100:.0f}%)")
print(f"veg→built: {constructed.sum()}")
print(f"veg→bare: {grading.sum()}")
print(f"Confirmed grading (veg→bare→built): {confirmed.sum()}")
print(f"Direct (veg→built no bare): {constructed.sum() - confirmed.sum()}")