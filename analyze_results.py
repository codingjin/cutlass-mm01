#!/usr/bin/env python3
"""
Analyze CUTLASS auto-tuning results and find optimal configurations
"""

import csv
import sys
from collections import defaultdict

def analyze_results(csv_file):
    """Analyze tuning results from CSV file"""

    if not csv_file:
        print("Usage: python3 analyze_results.py results.csv")
        return

    configs = []

    try:
        with open(csv_file, 'r') as f:
            # Skip any header lines until we find the CSV header
            lines = f.readlines()
            csv_start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('tb_m,tb_n,tb_k'):
                    csv_start = i
                    break

            # Parse CSV data starting from the header
            if csv_start > 0:
                csv_data = ''.join(lines[csv_start:])
            else:
                csv_data = ''.join(lines)

            reader = csv.DictReader(csv_data.splitlines())
            for row in reader:
                try:
                    config = {
                        'tb_m': int(row['tb_m']),
                        'tb_n': int(row['tb_n']),
                        'tb_k': int(row['tb_k']),
                        'warp_m': int(row['warp_m']),
                        'warp_n': int(row['warp_n']),
                        'warp_k': int(row['warp_k']),
                        'stages': int(row['stages']),
                        'time_ms': float(row['time_ms']),
                        'gflops': float(row['gflops'])
                    }
                    configs.append(config)
                except (ValueError, KeyError) as e:
                    continue

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        return

    if not configs:
        print("No valid configurations found in results")
        return

    # Sort by performance
    configs.sort(key=lambda x: x['gflops'], reverse=True)

    print("=" * 80)
    print("CUTLASS AUTO-TUNING RESULTS ANALYSIS")
    print("=" * 80)
    print(f"\nTotal configurations tested: {len(configs)}")
    print(f"Matrix size: 4096x4096 x 4096x4096 (float32)")
    print(f"GPU: RTX 3090 (Ampere SM86, TF32 Tensor Cores)")
    print()

    # Top 10 configurations
    print("=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Rank':<5} {'Threadblock':<15} {'Warp':<15} {'Stages':<7} {'Time(ms)':<10} {'GFLOPS':<10} {'Efficiency'}")
    print("-" * 80)

    for i, config in enumerate(configs[:10], 1):
        tb = f"{config['tb_m']}x{config['tb_n']}x{config['tb_k']}"
        warp = f"{config['warp_m']}x{config['warp_n']}x{config['warp_k']}"
        efficiency = (config['gflops'] / 71000.0) * 100
        print(f"{i:<5} {tb:<15} {warp:<15} {config['stages']:<7} "
              f"{config['time_ms']:<10.3f} {config['gflops']:<10.2f} {efficiency:.1f}%")

    # Best configuration
    best = configs[0]
    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATION")
    print("=" * 80)
    print(f"Threadblock Shape: {best['tb_m']} x {best['tb_n']} x {best['tb_k']}")
    print(f"Warp Shape:        {best['warp_m']} x {best['warp_n']} x {best['warp_k']}")
    print(f"Pipeline Stages:   {best['stages']}")
    print(f"Execution Time:    {best['time_ms']:.3f} ms")
    print(f"Performance:       {best['gflops']:.2f} GFLOPS")
    print(f"Theoretical Peak:  71000 GFLOPS (TF32)")
    print(f"Efficiency:        {(best['gflops'] / 71000.0) * 100:.1f}%")

    # Statistics by stages
    print("\n" + "=" * 80)
    print("PERFORMANCE BY PIPELINE STAGES")
    print("=" * 80)
    by_stages = defaultdict(list)
    for config in configs:
        by_stages[config['stages']].append(config['gflops'])

    for stages in sorted(by_stages.keys()):
        perfs = by_stages[stages]
        avg_gflops = sum(perfs) / len(perfs)
        max_gflops = max(perfs)
        print(f"Stages {stages}: Avg={avg_gflops:7.2f} GFLOPS, "
              f"Max={max_gflops:7.2f} GFLOPS, Configs={len(perfs)}")

    # Statistics by threadblock size
    print("\n" + "=" * 80)
    print("PERFORMANCE BY THREADBLOCK SIZE")
    print("=" * 80)
    by_tb = defaultdict(list)
    for config in configs:
        tb_key = f"{config['tb_m']}x{config['tb_n']}x{config['tb_k']}"
        by_tb[tb_key].append(config['gflops'])

    tb_stats = [(tb, max(perfs), sum(perfs)/len(perfs))
                for tb, perfs in by_tb.items()]
    tb_stats.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Threadblock':<15} {'Max GFLOPS':<12} {'Avg GFLOPS':<12} {'Configs'}")
    print("-" * 80)
    for tb, max_gf, avg_gf in tb_stats[:10]:
        print(f"{tb:<15} {max_gf:<12.2f} {avg_gf:<12.2f} {len(by_tb[tb])}")

    # Code generation
    print("\n" + "=" * 80)
    print("COPY-PASTE OPTIMAL CONFIGURATION")
    print("=" * 80)
    print(f"""
template<int TBM={best['tb_m']}, int TBN={best['tb_n']}, int TBK={best['tb_k']},
         int WM={best['warp_m']}, int WN={best['warp_n']}, int WK={best['warp_k']},
         int Stages={best['stages']}>
struct OptimalMatMulConfig {{
    using ThreadblockShape = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    static constexpr int kStages = Stages;
    // ... rest of configuration
}};
""")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    analyze_results(csv_file)
