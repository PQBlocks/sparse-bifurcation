#!/usr/bin/env python3
"""CLI Entry point for Sparse Structure-Aware Discrete Simulated Bifurcation."""

import argparse
import json
import sys
import os
import numpy as np

# Ensure workspace root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig


def parse_array(arg: str) -> np.ndarray:
    """Parses a JSON string or file path into a NumPy array."""
    if os.path.exists(arg):
        if arg.endswith(".npy"):
            return np.load(arg)
        with open(arg, "r") as f:
            return np.array(json.load(f), dtype=np.float64)
    return np.array(json.loads(arg), dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Run Sparse Structure-Aware dSB solver.")
    parser.add_argument("--j-matrix", required=True, help="Coupling matrix J (JSON string or file path)")
    parser.add_argument("--h-vector", default=None, help="Linear field vector h (JSON string or file path)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sweeps", "--num-sweeps", type=int, default=1000, dest="num_sweeps", help="Number of integration sweeps")
    parser.add_argument("--reads", "--num-reads", type=int, default=1, dest="num_reads", help="Number of reads")
    parser.add_argument("--dt", type=float, default=0.5, help="Symplectic Euler time step")
    parser.add_argument("--c0", type=float, default=None, help="Coupling constant override")
    parser.add_argument("--execution-mode", type=str, default="auto", choices=["auto", "dense", "csr", "locality_csr"], help="Execution mode")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")

    args = parser.parse_args()

    J = parse_array(args.j_matrix)
    N = J.shape[0]

    if args.h_vector is not None:
        h = parse_array(args.h_vector)
    else:
        h = np.zeros(N, dtype=np.float64)

    solver = SparseStructureAwareDSB(
        num_sweeps=args.num_sweeps,
        num_reads=args.num_reads,
        dt=args.dt,
        c0=args.c0,
        seed=args.seed,
        execution_mode=args.execution_mode
    )

    sampleset, receipt = solver.solve_with_receipt(h, J)

    best_s = sampleset.first.sample if hasattr(sampleset.first, "sample") else sampleset.record.sample[0]
    if isinstance(best_s, dict):
        best_spin_list = [int(best_s[i]) for i in range(N)]
    elif hasattr(best_s, "tolist"):
        best_spin_list = [int(x) for x in best_s.tolist()]
    else:
        best_spin_list = [int(x) for x in best_s]

    output_data = {
        "spins": [[int(x) for x in row] for row in sampleset.record.sample],
        "energies": [float(e) for e in sampleset.record.energy],
        "best_spin": best_spin_list,
        "best_energy": float(sampleset.first.energy),
        "metadata": {
            "execution_mode": str(receipt.execution_mode),
            "receipt": {
                "golden_id": str(receipt.golden_id),
                "input_sha256": str(receipt.input_sha256),
                "spin_sha256": str(receipt.spin_sha256),
                "energy": float(receipt.energy),
                "execution_mode": str(receipt.execution_mode),
                "seed": int(receipt.seed),
                "layout_cache_hit": bool(receipt.layout_cache_hit)
            }
        }
    }

    json_str = json.dumps(output_data, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
