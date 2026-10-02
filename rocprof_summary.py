#!/usr/bin/env python3
"""
Parse rocprofv3 CSV output and print a kernel summary table with optional
roofline analysis.

Usage:
    python3 rocprof_summary.py <directory>            # auto-detects newest run
    python3 rocprof_summary.py <directory> <pid>      # selects a specific run by PID
    python3 rocprof_summary.py <directory> --all      # summarises every run found
    python3 rocprof_summary.py <directory> --top N    # show top N kernels (default 20)

Output columns (kernel_trace.csv):
    Dispatches  — number of times the kernel was launched
    Total ms    — total GPU wall-clock time across all dispatches
    Avg µs      — mean duration per dispatch
    % GPU       — share of total GPU kernel time

Output columns (counter_collection.csv, if present):
    Occupancy%  — mean OccupancyPercent across dispatches
    MemBusy%    — mean MemUnitBusy across dispatches
    GPUBusy%    — mean GPUBusy across dispatches

Roofline columns (requires FETCH_SIZE, WRITE_SIZE, VALUInsts, Wavefronts in --pmc):
    DRAM GB/s   — achieved DRAM read+write bandwidth per kernel
    BW util%    — achieved / peak DRAM bandwidth
    VALU GFLOPS — lower-bound FLOP rate (VALUInsts × wave_size × 2 FLOPs)
    AI (F/B)    — arithmetic intensity in FLOP/byte vs DRAM traffic
    Bound       — MEMORY or COMPUTE relative to ridge point

Collect roofline counters with:
    rocprofv3 --kernel-trace \\
      --pmc GPUBusy OccupancyPercent MemUnitBusy WriteUnitStalled \\
             FETCH_SIZE WRITE_SIZE VALUInsts Wavefronts L2CacheHit \\
      -d <output_dir> -f csv -- python3 main.py validate ...

Hardware constants (Radeon 8060S / gfx1151):
    Peak FP16 compute : ~238 TFLOPS  (40 CU × 1024 MACs/CU/cycle × 2900 MHz × 2)
    Peak DRAM BW      : ~273 GB/s    (256-bit LPDDR5X @ 8533 Mbps)
    Ridge point       : ~870 FLOP/byte
"""

import csv
import glob
import os
import statistics
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Hardware constants — Radeon 8060S / gfx1151
# ---------------------------------------------------------------------------
# Radeon 8060S (RDNA 3.5, 40 CUs, 2900 MHz boost)
# FP16 theoretical peak via WMMA: 512 FP16 ops/cycle/CU × 40 × 2.9 GHz = 59.4 TFLOPS
# FP16 practical peak (hipBLASLt, measured): ~36.9 TFLOPS
# The earlier figure of 237.6 TFLOPS was derived incorrectly by multiplying raw MFMA
# FLOPs without accounting for the multi-cycle latency of MFMA instructions.
PEAK_TFLOPS_FP16  = 59.4           # theoretical peak FP16 WMMA (TFLOPS)
# Ryzen AI MAX+ 395 uses LPDDR5X-8000 on a 256-bit bus → 256 GB/s theoretical peak.
# (273 GB/s would be LPDDR5X-8533, used in the later PRO 400 / Gorgon Halo series.)
# Real-world bandwidth under mixed CPU+GPU load is typically ~212 GB/s.
PEAK_BW_GBS       = 256.0          # theoretical peak DRAM bandwidth (GB/s)
RIDGE_POINT       = (PEAK_TFLOPS_FP16 * 1e12) / (PEAK_BW_GBS * 1e9)  # FLOP/byte
WAVE_SIZE         = 32             # Wave32 on RDNA4
FLOPS_PER_VALU    = 2 * WAVE_SIZE  # FP16 MAD × 32 lanes (lower bound; MFMA does more)
MAX_VGPR_PER_THREAD = 256          # RDNA4: 256 VGPRs per thread (per lane)
MAX_WAVES_PER_SIMD  = 16           # RDNA4: max 16 wavefronts per SIMD unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_runs(directory):
    """Return a dict mapping pid -> {'kernel_trace': path, 'counters': [path, ...]}."""
    pattern = os.path.join(directory, "*_kernel_trace.csv")
    runs = {}
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)
        pid = basename.split("_kernel_trace.csv")[0]
        counter_path = os.path.join(directory, f"{pid}_counter_collection.csv")
        runs[pid] = {
            "kernel_trace": path,
            "counters": [counter_path] if os.path.exists(counter_path) else [],
        }
    return runs


def parse_kernel_trace(path):
    """
    Read *_kernel_trace.csv.

    Returns a dict keyed by Kernel_Name containing:
        count     : int   — number of dispatches
        total_ns  : int   — total GPU time in nanoseconds
        grids     : list  — per-dispatch total grid size (X*Y*Z)
        vgpr      : int   — VGPR count per thread (first observed value; constant per kernel)
        sgpr      : int   — SGPR count per wavefront
        lds_kb    : float — LDS block size in KB

    GPU utilisation over the full profiling window is stored under
    the special key "__utilisation__".
    """
    kernels = defaultdict(lambda: {"count": 0, "total_ns": 0, "grids": [],
                                   "vgpr": None, "sgpr": None, "lds_kb": None})
    intervals = []

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name  = row["Kernel_Name"]
            start = int(row["Start_Timestamp"])
            end   = int(row["End_Timestamp"])
            dur   = end - start
            grid  = (
                int(row["Grid_Size_X"])
                * int(row["Grid_Size_Y"])
                * int(row["Grid_Size_Z"])
            )
            kernels[name]["count"]    += 1
            kernels[name]["total_ns"] += dur
            kernels[name]["grids"].append(grid)
            intervals.append((start, end))
            # VGPR/SGPR/LDS are constant per kernel — capture once
            if kernels[name]["vgpr"] is None:
                kernels[name]["vgpr"]   = int(row.get("VGPR_Count", 0) or 0)
                kernels[name]["sgpr"]   = int(row.get("SGPR_Count",  0) or 0)
                kernels[name]["lds_kb"] = int(row.get("LDS_Block_Size", 0) or 0) / 1024

    # GPU utilisation: merge overlapping intervals
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append([start, end])

    timeline_span_ns = merged[-1][1] - merged[0][0] if merged else 0
    active_gpu_ns    = sum(e - s for s, e in merged)
    idle_ns          = timeline_span_ns - active_gpu_ns
    utilisation_pct  = 100.0 * active_gpu_ns / timeline_span_ns if timeline_span_ns else 0.0

    kernels["__utilisation__"] = {
        "timeline_span_ns": timeline_span_ns,
        "active_gpu_ns":    active_gpu_ns,
        "idle_ns":          idle_ns,
        "utilisation_pct":  utilisation_pct,
    }

    return kernels


def parse_counter_collection(path):
    """
    Read one *_counter_collection.csv.

    Returns a nested dict:
        counters[Kernel_Name][Counter_Name] -> list of float values
    """
    counters = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            counters[row["Kernel_Name"]][row["Counter_Name"]].append(
                float(row["Counter_Value"])
            )
    return counters


def merge_counter_collections(paths):
    """
    Read and merge one or more *_counter_collection.csv files.

    When the hardware PMU cannot collect all desired counters in a single pass,
    run rocprofv3 twice with different --pmc subsets and pass both resulting
    counter CSV files here.  Counters are merged by Kernel_Name; duplicate
    counter names from multiple files are concatenated (the mean is taken later).
    """
    merged = defaultdict(lambda: defaultdict(list))
    for path in paths:
        partial = parse_counter_collection(path)
        for kernel, cmap in partial.items():
            for counter, values in cmap.items():
                merged[kernel][counter].extend(values)
    return merged


def mean(lst):
    return statistics.mean(lst) if lst else None


def has_roofline_counters(counters):
    """
    Return True if enough counters are present in any kernel's data to compute
    roofline metrics.

    Memory traffic source (in priority order):
      1. FETCH_SIZE + WRITE_SIZE  (derived KB counters — NOT available on gfx1151 iGPU)
      2. GL2C_MC_RDREQ_sum + GL2C_MC_WRREQ_sum  (raw L2→DRAM request counts × 64B)

    Compute source (either accepted):
      VALUInsts + Wavefronts  OR  VALUInsts + SQ_WAVES_sum
    """
    for c in counters.values():
        has_mem = (
            ("FETCH_SIZE" in c and "WRITE_SIZE" in c) or
            ("GL2C_MC_RDREQ_sum" in c and "GL2C_MC_WRREQ_sum" in c)
        )
        has_compute = (
            "VALUInsts" in c and
            ("Wavefronts" in c or "SQ_WAVES_sum" in c)
        )
        if has_mem and has_compute:
            return True
    return False


# Cache line size assumed for GL2C request-count → byte conversion
GL2C_CACHE_LINE_BYTES = 64


def roofline_metrics(c, total_ns):
    """
    Compute roofline metrics for one kernel.

    Memory traffic:
      Preferred: (FETCH_SIZE + WRITE_SIZE) × 1024 bytes  [derived, KB]
      Fallback:  (GL2C_MC_RDREQ_sum + GL2C_MC_WRREQ_sum) × 64 bytes
                 (each MC request fetches one 64-byte cache line on average)

    Compute (lower bound):
      VALUInsts (avg per work-item) × (Wavefronts | SQ_WAVES_sum) × 32 lanes × 2 FLOPs
      Note: MFMA instructions count as one VALU instruction but execute 8192 FLOPs
            (MI16×16×16), so GEMM kernels are actually far above the computed value.

    L2 hit rate (optional, shown separately):
      GL2C_HIT_sum / (GL2C_HIT_sum + GL2C_MISS_sum)  ×  100

    Returns a dict or None if required counters are absent.
    """
    # --- memory traffic ---
    if "FETCH_SIZE" in c and "WRITE_SIZE" in c:
        total_bytes  = (mean(c["FETCH_SIZE"]) + mean(c["WRITE_SIZE"])) * 1024
        mem_src      = "KB"
    elif "GL2C_MC_RDREQ_sum" in c and "GL2C_MC_WRREQ_sum" in c:
        rd = mean(c["GL2C_MC_RDREQ_sum"])
        wr = mean(c["GL2C_MC_WRREQ_sum"])
        if rd is None or wr is None:
            return None
        total_bytes = (rd + wr) * GL2C_CACHE_LINE_BYTES
        mem_src     = "GL2C"
    else:
        return None

    if total_bytes is None or total_bytes == 0:
        return None

    # --- compute ---
    valu  = mean(c.get("VALUInsts", []))
    waves = mean(c.get("Wavefronts", c.get("SQ_WAVES_sum", [])))
    if valu is None or waves is None:
        return None

    # --- L2 hit rate (optional) ---
    l2_hit  = None
    hit_v   = mean(c.get("GL2C_HIT_sum",  []) or c.get("L2CacheHit", []))
    miss_v  = mean(c.get("GL2C_MISS_sum", []))
    if hit_v is not None and miss_v is not None and (hit_v + miss_v) > 0:
        l2_hit = 100.0 * hit_v / (hit_v + miss_v)
    elif hit_v is not None and "L2CacheHit" in c:
        l2_hit = hit_v  # already a percentage

    # --- derived values ---
    duration_s       = total_ns / 1e9
    dram_gbs         = (total_bytes / 1e9) / duration_s if duration_s > 0 else 0.0
    bw_util_pct      = 100.0 * dram_gbs / PEAK_BW_GBS
    total_valu_insts = valu * waves * WAVE_SIZE
    gflops_rate      = total_valu_insts * FLOPS_PER_VALU / 1e9 / duration_s if duration_s > 0 else 0.0
    ai               = (total_valu_insts * FLOPS_PER_VALU) / total_bytes

    return {
        "dram_gbs":    dram_gbs,
        "bw_util_pct": bw_util_pct,
        "gflops":      gflops_rate,
        "ai":          ai,
        "l2_hit":      l2_hit,
        "mem_src":     mem_src,
        "bound":       "COMPUTE*" if ai > RIDGE_POINT else "MEMORY",
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

DIVIDER_WIDTH = 145

def print_summary(pid, kernel_trace_path, counter_paths, top_n=20):
    """
    counter_paths : list of paths to *_counter_collection.csv files (may be empty).
    Multiple files are merged so that counters collected in separate PMU passes
    are combined into one view.
    """
    print("=" * DIVIDER_WIDTH)
    print(f"Run PID : {pid}")
    print(f"Trace   : {kernel_trace_path}")
    for p in counter_paths:
        print(f"Counters: {p}")
    print("=" * DIVIDER_WIDTH)

    kernels  = parse_kernel_trace(kernel_trace_path)
    counters = merge_counter_collections(counter_paths) if counter_paths else {}

    util             = kernels.pop("__utilisation__")
    total_gpu_ns     = sum(v["total_ns"] for v in kernels.values())
    total_dispatches = sum(v["count"]    for v in kernels.values())

    print(
        f"Total GPU kernel time : {total_gpu_ns / 1e9:.3f} s  |  "
        f"Total dispatches : {total_dispatches:,}  |  "
        f"Unique kernels : {len(kernels)}"
    )
    print(
        f"Profiling window      : {util['timeline_span_ns'] / 1e9:.3f} s  |  "
        f"GPU active : {util['active_gpu_ns'] / 1e9:.3f} s  |  "
        f"GPU idle : {util['idle_ns'] / 1e9:.3f} s  |  "
        f"GPU utilisation : {util['utilisation_pct']:.1f}%"
    )
    print(
        f"  → '% GPU' column = share of the {total_gpu_ns/1e9:.3f} s active window; "
        f"GPU is idle the remaining {util['idle_ns']/1e9:.3f} s "
        f"(CPU postproc, data loading, Python dispatch overhead)"
    )

    do_roofline = has_roofline_counters(counters)
    if do_roofline:
        print(
            f"Roofline hardware     : ridge = {RIDGE_POINT:.0f} FLOP/byte  |  "
            f"peak BW = {PEAK_BW_GBS:.0f} GB/s  |  "
            f"peak compute = {PEAK_TFLOPS_FP16:.0f} TFLOPS FP16\n"
            f"  → 'COMPUTE*' means AI > ridge using VALU lower bound; "
            f"MFMA instructions do ~128× more FLOP/inst than regular VALU,\n"
            f"    so GEMM kernels are almost certainly compute-bound in practice."
        )
    print()

    ranked = sorted(kernels.items(), key=lambda x: -x[1]["total_ns"])

    has_occ = bool(counters)

    # Build header
    base = f"{'#':>3}  {'Kernel (60 chars)':<62} {'Dispatches':>10} {'Total ms':>10} {'Avg µs':>9} {'% GPU':>7} {'Avg grid':>10} {'VGPR':>5} {'MaxW':>5} {'OccCeil':>8}"
    occ_part = f" {'Occupancy%':>11} {'MemBusy%':>9} {'GPUBusy%':>9}" if has_occ else ""
    rl_part  = f" {'DRAM GB/s':>10} {'BW util%':>9} {'VFLOPS':>8} {'AI F/B':>8} {'L2Hit%':>7} {'Bound':>9}" if do_roofline else ""
    header   = base + occ_part + rl_part
    print(header)
    print("-" * len(header))

    for rank, (name, v) in enumerate(ranked[:top_n], start=1):
        avg_grid = mean(v["grids"])
        total_ms = v["total_ns"] / 1e6
        avg_us   = v["total_ns"] / v["count"] / 1e3
        pct      = 100.0 * v["total_ns"] / total_gpu_ns

        vgpr     = v.get("vgpr") or 0
        max_w    = (MAX_VGPR_PER_THREAD // vgpr) if vgpr > 0 else 0
        occ_ceil = min(max_w, MAX_WAVES_PER_SIMD) / MAX_WAVES_PER_SIMD * 100 if max_w > 0 else 0
        line = (
            f"{rank:>3}  {name[:60]:<62} {v['count']:>10,} {total_ms:>10.1f} "
            f"{avg_us:>9.1f} {pct:>6.1f}% {avg_grid:>10,.0f} "
            f"{vgpr:>5} {max_w:>5} {occ_ceil:>7.0f}%"
        )

        if has_occ:
            c   = counters.get(name, {})
            occ = mean(c.get("OccupancyPercent", []))
            mem = mean(c.get("MemUnitBusy",      []))
            gpu = mean(c.get("GPUBusy",           []))
            line += (
                f" {occ:>11.1f}" if occ is not None else f" {'—':>11}"
            ) + (
                f" {mem:>9.1f}" if mem is not None else f" {'—':>9}"
            ) + (
                f" {gpu:>9.1f}" if gpu is not None else f" {'—':>9}"
            )

            if do_roofline:
                rl = roofline_metrics(c, v["total_ns"])
                if rl:
                    l2_s = f"{rl['l2_hit']:>7.1f}" if rl["l2_hit"] is not None else f"{'—':>7}"
                    line += (
                        f" {rl['dram_gbs']:>10.1f}"
                        f" {rl['bw_util_pct']:>9.1f}"
                        f" {rl['gflops']:>8.1f}"
                        f" {rl['ai']:>8.1f}"
                        f" {l2_s}"
                        f" {rl['bound']:>9}"
                    )
                else:
                    line += f" {'—':>10} {'—':>9} {'—':>8} {'—':>8} {'—':>7} {'—':>9}"

        print(line)

    if len(ranked) > top_n:
        remaining_ns  = sum(v["total_ns"] for _, v in ranked[top_n:])
        remaining_cnt = sum(v["count"]    for _, v in ranked[top_n:])
        remaining_pct = 100.0 * remaining_ns / total_gpu_ns
        print(
            f"{'':>3}  {'... ' + str(len(ranked) - top_n) + ' more kernels':<62} "
            f"{remaining_cnt:>10,} {remaining_ns/1e6:>10.1f} {'':>9} "
            f"{remaining_pct:>6.1f}%"
        )

    if do_roofline:
        mem_src_note = (
            "GL2C_MC_RDREQ/WRREQ_sum × 64B (raw L2→DRAM requests)"
            if any("GL2C_MC_RDREQ_sum" in c for c in counters.values())
            else "FETCH_SIZE+WRITE_SIZE KB (derived)"
        )
        print()
        print(f"  Roofline: DRAM traffic = {mem_src_note}; "
              f"AI = VALU lower-bound FLOP / DRAM bytes; "
              f"ridge = {RIDGE_POINT:.0f} FLOP/byte; "
              f"COMPUTE* = AI > ridge (GEMM kernels are much higher due to MFMA)")

    print()

    # Return roofline data for optional plotting.
    # rank matches the # column in the printed table (1-based, sorted by GPU time).
    if do_roofline:
        rl_data = []
        for rank, (name, v) in enumerate(ranked, start=1):
            c  = counters.get(name, {})
            rl = roofline_metrics(c, v["total_ns"])
            if rl and rl["ai"] > 0 and rl["gflops"] > 0:
                rl_data.append({
                    "rank":    rank,
                    "name":    name,
                    "ai":      rl["ai"],
                    "gflops":  rl["gflops"],
                    "pct_gpu": 100.0 * v["total_ns"] / total_gpu_ns,
                    "bound":   rl["bound"],
                })
        return rl_data
    return []


def plot_roofline(rl_points, output_path):
    """
    Generate a roofline scatter plot and save to output_path (.png or .pdf).

    X axis : arithmetic intensity (FLOP/byte), log scale
    Y axis : achieved GFLOPS/s, log scale
    Lines  : memory-bandwidth roof and compute roof
    Points : one per kernel, sized by % GPU time, coloured by bound regime
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plot. Install with: pip install matplotlib",
              file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Roofline hardware limits
    ai_range = np.logspace(-2, 5, 500)
    mem_roof     = PEAK_BW_GBS  * ai_range        # memory-bandwidth ceiling (GFLOPS/s)
    compute_roof = np.full_like(ai_range, PEAK_TFLOPS_FP16 * 1e3)  # compute ceiling

    roof = np.minimum(mem_roof, compute_roof)
    ax.loglog(ai_range, roof, "k-", linewidth=2, label="Hardware roofline")
    ax.axvline(RIDGE_POINT, color="grey", linestyle="--", linewidth=1,
               label=f"Ridge point ({RIDGE_POINT:.0f} FLOP/byte)")

    # Annotation for memory and compute roofs
    ax.text(ai_range[50],  mem_roof[50] * 1.3,
            f"Memory BW ({PEAK_BW_GBS:.0f} GB/s)",
            fontsize=8, color="black", rotation=30)
    ax.text(ai_range[-80], PEAK_TFLOPS_FP16 * 1e3 * 0.7,
            f"Compute ({PEAK_TFLOPS_FP16:.0f} TFLOPS FP16)",
            fontsize=8, color="black")

    # Kernel points — plot all, but only label the top 20 by GPU time
    colours  = {"MEMORY": "#e07030", "COMPUTE*": "#3070d0"}
    top20    = set(pt["rank"] for pt in sorted(rl_points, key=lambda p: p["rank"])[:20])
    for pt in rl_points:
        colour = colours.get(pt["bound"], "#888888")
        size   = max(30, min(600, pt["pct_gpu"] * 30))
        ax.scatter(pt["ai"], pt["gflops"], s=size, color=colour, alpha=0.75,
                   edgecolors="white", linewidths=0.5, zorder=3)
        if pt["rank"] in top20:
            rank  = pt["rank"]
            short = pt["name"].split("_")[0][:18]
            label = f"#{rank} {short}\n({pt['pct_gpu']:.1f}%)"
            ax.annotate(label,
                        (pt["ai"], pt["gflops"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, color=colour)

    # Legend proxies for kernel colours
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=colours["MEMORY"],   label="Memory-bound (AI < ridge)"),
        mpatches.Patch(color=colours["COMPUTE*"], label="Compute-bound* (AI > ridge, VALU LB)"),
    ]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0][:-1],
              loc="lower right", fontsize=8)

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=11)
    ax.set_ylabel("Achieved Performance (GFLOPS/s)", fontsize=11)
    ax.set_title(
        f"Roofline Model — gfx1151 (Radeon 8060S)\n"
        f"Peak: {PEAK_TFLOPS_FP16:.0f} TFLOPS FP16  |  "
        f"Peak BW: {PEAK_BW_GBS:.0f} GB/s  |  "
        f"Ridge: {RIDGE_POINT:.0f} FLOP/byte  |  "
        f"Point size ∝ % GPU time",
        fontsize=10,
    )
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.set_xlim(1e-1, 1e4)
    ax.set_ylim(1, PEAK_TFLOPS_FP16 * 1e3 * 3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Roofline plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def newest_run(directory):
    """
    Return the single latest run from *directory* as a dict with keys
    'pid', 'kernel_trace', and 'counters'.  Raises SystemExit if none found.
    """
    runs = find_runs(directory)
    if not runs:
        print(f"No *_kernel_trace.csv files found in: {directory}", file=sys.stderr)
        sys.exit(1)
    pid = max(runs, key=lambda p: int(p))
    return {"pid": pid, **runs[pid]}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <dir> [<dir2> ...] [--all] [--top N] [--extra-counters <csv>] [--plot [file.png]]")
        print()
        print("  Single-directory mode (original):")
        print("    <dir>                  folder containing rocprofv3 CSV files")
        print("    <pid>                  (optional) select a specific run by PID; default: newest")
        print("    --all                  summarise every run found in the directory")
        print()
        print("  Multi-directory mode (two-pass roofline, no PID needed):")
        print("    <dir1> <dir2> ...      pass one or more directories; the newest run from each")
        print("                           is selected automatically and their counters are merged.")
        print("    Example:")
        print("      python3 rocprof_summary.py \\")
        print("        rocprof_output_roofline_pass1/xcoradaie217 \\")
        print("        rocprof_output_roofline_pass2/xcoradaie217")
        print()
        print("  Shared options:")
        print("    --top N                show top N kernels (default: 20)")
        print("    --extra-counters F     merge an additional counter CSV")
        print("    --plot [file.png]      save a roofline scatter plot (default: roofline.png)")
        sys.exit(1)

    args = sys.argv[1:]

    top_n = 20
    if "--top" in args:
        idx   = args.index("--top")
        top_n = int(args[idx + 1])
        args  = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    plot_path = None
    if "--plot" in args:
        idx       = args.index("--plot")
        plot_path = args[idx + 1] if idx + 1 < len(args) and not args[idx + 1].startswith("--") \
                    else "roofline.png"
        args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    extra_counters = []
    # Accept both --extra-counters (canonical) and --extra-counter (common typo)
    for flag in ("--extra-counters", "--extra-counter"):
        while flag in args:
            idx  = args.index(flag)
            path = args[idx + 1]
            if not os.path.exists(path):
                print(f"{flag} file not found: {path}", file=sys.stderr)
                sys.exit(1)
            extra_counters.append(path)
            args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    want_all = "--all" in args
    if want_all:
        args = [a for a in args if a != "--all"]

    # Separate positional directory arguments from any remaining flags
    dirs = [a for a in args if not a.startswith("--") and os.path.isdir(a)]
    non_dirs = [a for a in args if not a.startswith("--") and not os.path.isdir(a)]

    all_rl_points = []

    if len(dirs) > 1:
        # ---------------------------------------------------------------
        # Multi-directory mode: one newest run per directory, merged.
        # ---------------------------------------------------------------
        if want_all or non_dirs:
            print(
                "Warning: --all and explicit PIDs are ignored in multi-directory mode "
                "(newest run per directory is always used).",
                file=sys.stderr,
            )
        runs_list = [newest_run(d) for d in dirs]
        # The kernel trace comes from the first directory that has one.
        trace_run = next((r for r in runs_list if r["kernel_trace"]), None)
        if trace_run is None:
            print("No kernel trace found in any of the supplied directories.", file=sys.stderr)
            sys.exit(1)
        counter_paths = []
        for r in runs_list:
            counter_paths.extend(r["counters"])
        counter_paths.extend(extra_counters)
        label = " + ".join(dirs)
        rl_points = print_summary(label, trace_run["kernel_trace"], counter_paths, top_n=top_n)
        all_rl_points.extend(rl_points)
    else:
        # ---------------------------------------------------------------
        # Single-directory mode (original behaviour).
        # ---------------------------------------------------------------
        directory = dirs[0] if dirs else (args[0] if args else None)
        if directory is None or not os.path.isdir(directory):
            print(f"Directory not found: {directory}", file=sys.stderr)
            sys.exit(1)

        runs = find_runs(directory)
        if not runs:
            print(f"No *_kernel_trace.csv files found in: {directory}", file=sys.stderr)
            sys.exit(1)

        if want_all:
            selected = runs
        elif non_dirs:
            pid = non_dirs[0]
            if pid not in runs:
                print(f"PID '{pid}' not found. Available: {', '.join(sorted(runs))}", file=sys.stderr)
                sys.exit(1)
            selected = {pid: runs[pid]}
        else:
            pid      = max(runs, key=lambda p: int(p))
            selected = {pid: runs[pid]}

        for pid, paths in sorted(selected.items(), key=lambda x: int(x[0])):
            counter_paths = paths["counters"] + extra_counters
            rl_points = print_summary(pid, paths["kernel_trace"], counter_paths, top_n=top_n)
            all_rl_points.extend(rl_points)

    if plot_path and all_rl_points:
        plot_roofline(all_rl_points, plot_path)
    elif plot_path:
        print("No roofline data available for plotting — ensure roofline counters were collected.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
