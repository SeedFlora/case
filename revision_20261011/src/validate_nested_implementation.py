"""Scientific correctness checks for the revised experimental driver."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nested_reanalysis as n
import numpy as np
import torch

torch.set_num_threads(1)
cfg = n.load_configs()
reference = n.initialized_net("gated", 8, 10, cfg, 13000)
common = {k:v.clone() for k,v in reference.state_dict().items() if k.startswith(("encoders.", "heads."))}
checks = {}
for name, spec in n.NEURAL_SPECS.items():
    if spec["cls"] == "early":
        continue
    model = n.initialized_net(spec["cls"], 8, 10, cfg, 13000, common, **spec["kw"])
    checks["shared_initialization_"+name] = all(torch.equal(model.state_dict()[k], v) for k,v in common.items())
equal = n.initialized_net("gated", 8, 10, cfg, 13000, common, learned_gates=False).eval()
late = n.initialized_net("late", 8, 10, cfg, 13000, common).eval()
rng = np.random.RandomState(123)
x = torch.from_numpy(rng.normal(size=(7, 8, 320)).astype(np.float32))
q = torch.from_numpy(rng.normal(size=(7, 8, 10)).astype(np.float32))
mask = torch.ones(7, 8)
mask[:, (0, 3, 6)] = 0
with torch.no_grad():
    checks["equal_late_forward_identical"] = torch.equal(equal(x,q,mask), late(x,q,mask))
xx = rng.normal(size=(9, 8, 320)).astype(np.float32)
channels = np.array([0, 2, 4, 6])
checks["quality_channel_subset_equivalent"] = np.array_equal(n.compute_quality(xx)[:,channels], n.compute_quality(xx[:,channels,:]))
checks["features_channel_subset_equivalent"] = np.array_equal(n.features.extract(xx).reshape(9,8,-1)[:,channels], n.features.extract(xx[:,channels,:]).reshape(9,4,-1))
for w in n.WINDOWS:
    _, _, original_y, meta = n.load_windows(w, 0.)
    y, keep, coverage = n.candidate_labels(meta, w, 0., original_y)
    checks[f"zero_delay_label_equivalence_w{w:g}"] = np.array_equal(original_y, y)
    checks[f"zero_delay_no_new_exclusions_w{w:g}"] = bool(keep.all())
checks["all_passed"] = all(checks.values())
n.write_json(checks, n.REV/"manifests"/"implementation_validation.json")
print(checks)
assert checks["all_passed"]
