"""Joins a CUDA and a Vulkan run of `gemm_shapes_bench` into one table.

    cargo run --release --features cuda --example gemm_shapes_bench > cuda.txt
    cargo run --release --features vulkan --example gemm_shapes_bench > vulkan.txt
    python3 scripts/compare_gemm_shapes.py cuda.txt vulkan.txt

Rows more than 20% apart either way are flagged.
"""
import sys
def load(path):
    rows = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 8 and p[0] in ("NT", "NN", "Q8"):
            key = (p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4]))
            rows[key] = (int(p[5]), float(p[6]))
    return rows
cuda, vk = load(sys.argv[1]), load(sys.argv[2])
print(f"{'lay':<3} {'m':>5} {'n':>5} {'k':>5} {'b':>3} {'calls':>5} {'cuda us':>8} {'vk us':>8} {'vk/cuda':>8}  {'frame cuda':>10} {'frame vk':>9}")
fc = fv = 0.0
worst = []
for key in cuda:
    c, (calls, cu) = key, cuda[key]
    vu = vk[key][1]
    lay, m, n, k, b = key
    per_frame = 0 if (lay == "Q8" and m > 16) else calls
    fc += cu * per_frame; fv += vu * per_frame
    ratio = vu / cu
    flag = " <--" if ratio > 1.2 or ratio < 1 / 1.2 else ""
    print(f"{lay:<3} {m:>5} {n:>5} {k:>5} {b:>3} {calls:>5} {cu:>8.2f} {vu:>8.2f} {ratio:>8.2f}  {cu*per_frame:>10.1f} {vu*per_frame:>9.1f}{flag}")
print(f"per-frame matmul total: cuda {fc:.0f} us, vulkan {fv:.0f} us, ratio {fv/fc:.2f}")
