import numpy as np

x = np.load(
    "data/overlap_0/sdb4/windows_corrected.npz"
)

print(np.unique(x["cap"]))

print(x["cap"][0])
print(x["cap"][10])
print(x["cap"][50])