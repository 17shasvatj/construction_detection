import json, numpy as np

results = json.load(open('outputs/eval_results.json'))
tp = results['wendell']['per_timepoint']

# Aggregate confusion across timepoints, then collapse to 2-class
total_conf = np.zeros((3,3), dtype=np.int64)
for r in tp:
    total_conf += np.array(r['confusion'])

# Merge: class 0 = baseline, classes 1+2 = construction
# 2x2: [[base->base, base->constr], [constr->base, constr->constr]]
tn = total_conf[0,0]                          # baseline correct
fp = total_conf[0,1] + total_conf[0,2]        # baseline -> construction
fn = total_conf[1,0] + total_conf[2,0]        # construction -> baseline
tp_ = (total_conf[1,1] + total_conf[1,2] +    # construction -> construction
       total_conf[2,1] + total_conf[2,2])

prec = tp_ / (tp_ + fp + 1e-8)
rec  = tp_ / (tp_ + fn + 1e-8)
f1   = 2*prec*rec/(prec+rec+1e-8)
iou  = tp_ / (tp_ + fp + fn + 1e-8)
print(f"2-class CONSTRUCTION: P={prec:.3f} R={rec:.3f} F1={f1:.3f} IoU={iou:.3f}")