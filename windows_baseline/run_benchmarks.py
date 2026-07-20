#!/usr/bin/env python3
"""Run all quantum simulation benchmarks on Windows/WSL and aggregate results."""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

# Set matplotlib backend to non-interactive
os.environ["MPLBACKEND"] = "Agg"

def run_cmd(args):
    print(f"Executing: {' '.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with error:\n{result.stderr}")
    else:
        print(result.stdout.strip())
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", default="15,20,24,26,28")
    parser.add_argument("--qubits-default-qubit", default="15,20,24")
    parser.add_argument("--qubits-mps", default="15,20,24,26,28,30,40")
    parser.add_argument("--qubits-cudaq", default="15,20,24,26,28")
    parser.add_argument("--qubits-cudaq-cpu", default="15,20,24")
    parser.add_argument("--benchmarks", default="qft,qaoa_ring,ghz,grover_proxy,phase_estimation,tfim_trotter")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "results"
    plots_dir = base_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    print(f"Using python interpreter: {python_bin}")

    # Define tasks to run
    tasks = [
        # Qiskit Aer Statevector CPU
        [python_bin, str(base_dir / "qiskit_aer_baseline.py"), "--outdir", str(out_dir), "--method", "statevector", "--device", "CPU", "--qubits", args.qubits, "--benchmarks", args.benchmarks, "--warmups", str(args.warmups), "--repeats", str(args.repeats)],
        # Qiskit Aer Statevector GPU
        [python_bin, str(base_dir / "qiskit_aer_baseline.py"), "--outdir", str(out_dir), "--method", "statevector", "--device", "GPU", "--qubits", args.qubits, "--benchmarks", args.benchmarks, "--warmups", str(args.warmups), "--repeats", str(args.repeats)],
        # Qiskit Aer MPS CPU
        [python_bin, str(base_dir / "qiskit_aer_baseline.py"), "--outdir", str(out_dir), "--method", "matrix_product_state", "--device", "CPU", "--qubits", args.qubits_mps, "--benchmarks", args.benchmarks, "--warmups", str(args.warmups), "--repeats", str(args.repeats)],
        # Qiskit Aer MPS GPU
        [python_bin, str(base_dir / "qiskit_aer_baseline.py"), "--outdir", str(out_dir), "--method", "matrix_product_state", "--device", "GPU", "--qubits", args.qubits_mps, "--benchmarks", args.benchmarks, "--warmups", str(args.warmups), "--repeats", str(args.repeats)],
        
        # PennyLane default.qubit
        [python_bin, str(base_dir / "pennylane_baseline.py"), "--outdir", str(out_dir), "--device", "default.qubit", "--qubits", args.qubits_default_qubit, "--benchmarks", args.benchmarks, "--warmups", str(args.warmups), "--repeats", str(args.repeats)],
        # PennyLane lightning.qubit
        [python_bin, str(base_dir / "pennylane_baseline.py"), "--outdir", str(out_dir), "--device", "lightning.qubit", "--qubits", args.qubits, "--benchmarks", args.benchmarks, "--warmups", str(args.warmups), "--repeats", str(args.repeats)],
        # PennyLane lightning.gpu
        [python_bin, str(base_dir / "pennylane_baseline.py"), "--outdir", str(out_dir), "--device", "lightning.gpu", "--qubits", args.qubits, "--benchmarks", args.benchmarks, "--warmups", str(args.warmups), "--repeats", str(args.repeats)],
        
        # CUDA Quantum qpp (CPU)
        [python_bin, str(base_dir / "cudaq_baseline.py"), "--outdir", str(out_dir), "--backend", "qpp", "--qubits", args.qubits_cudaq_cpu, "--benchmarks", args.benchmarks, "--warmups", str(args.warmups), "--repeats", str(args.repeats)],
        # CUDA Quantum nvidia (GPU)
        [python_bin, str(base_dir / "cudaq_baseline.py"), "--outdir", str(out_dir), "--backend", "nvidia", "--qubits", args.qubits_cudaq, "--benchmarks", args.benchmarks, "--warmups", str(args.warmups), "--repeats", str(args.repeats)],
    ]

    print("--- Starting Benchmark Runs ---")
    for task in tasks:
        run_cmd(task)
    print("--- Completed Benchmark Runs ---")

    # Aggregate results
    print("--- Aggregating Results ---")
    summaries = list(out_dir.glob("*_summary.csv"))
    
    # Store data as {benchmark: {backend: {qubits: elapsed_ms}}}
    data = {}
    
    for summary_file in summaries:
        with summary_file.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                bench = row["benchmark"]
                backend = row["backend"]
                qubits = int(row["qubits"])
                mean_ms = float(row["mean_ms"])
                
                if bench not in data:
                    data[bench] = {}
                if backend not in data[bench]:
                    data[bench][backend] = {}
                data[bench][backend][qubits] = mean_ms

    # Generate Markdown report
    report_file = base_dir / "results_summary.md"
    report_lines = [
        "# Windows / WSL Baseline Performance Report",
        "",
        "This report aggregates simulation results for Qiskit Aer, PennyLane, and CUDA Quantum backends run on Windows WSL (Ubuntu) with an NVIDIA RTX 3070 8GB GPU.",
        "",
    ]
    
    for bench in sorted(data.keys()):
        report_lines.append(f"## Workload: {bench}")
        report_lines.append("")
        
        # Determine all qubits and backends for this benchmark
        all_backends = sorted(data[bench].keys())
        all_qubits = sorted(list(set(q for b in all_backends for q in data[bench][b].keys())))
        
        # Build table header
        header = "| Backend / Qubits | " + " | ".join(f"{q}q" for q in all_qubits) + " |"
        separator = "|---|" + "|---:|" * len(all_qubits)
        report_lines.append(header)
        report_lines.append(separator)
        
        # Build rows
        for backend in all_backends:
            row_cells = [backend]
            for q in all_qubits:
                val = data[bench][backend].get(q)
                if val is not None:
                    if val >= 1000.0:
                        row_cells.append(f"{val/1000.0:.3f} s")
                    else:
                        row_cells.append(f"{val:.2f} ms")
                else:
                    row_cells.append("-")
            report_lines.append("| " + " | ".join(row_cells) + " |")
        report_lines.append("")

    report_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Generated Markdown Summary: {report_file}")

    # Generate comparative plots
    try:
        import matplotlib.pyplot as plt
        print("--- Generating Comparative Plots ---")
        
        # Professional color palette and markers for each backend (matching exact CSV names)
        backend_styles = {
            "cudaq_nvidia": ("CUDA-Q GPU (RTX 3070)", "#5E9E00", "s", "-"),
            "qiskit_aer_statevector_gpu": ("Qiskit Aer GPU (RTX 3070)", "#E31A1C", "d", "-"),
            "qiskit_aer_statevector_cpu": ("Qiskit Aer CPU", "#FC9A99", "d", "--"),
            "lightning.gpu": ("PennyLane Lightning GPU (RTX 3070)", "#6A3D9A", "^", "-"),
            "lightning.qubit": ("PennyLane Lightning CPU", "#CAB2D6", "^", "--"),
            "default.qubit": ("PennyLane default.qubit (CPU)", "#FF7F00", "o", ":")
        }
        
        for bench in data.keys():
            fig, ax = plt.subplots(figsize=(9, 5.5))
            
            for backend in sorted(data[bench].keys()):
                # Exclude approximate matrix product state simulations
                if "matrix_product_state" in backend or "mps" in backend:
                    continue
                # Exclude slow CUDA-Q CPU simulation
                if "cudaq_qpp" in backend or "qpp" in backend:
                    continue
                
                points = sorted(data[bench][backend].items())
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                
                # Fetch professional styles, fallback to default if not configured
                label, color, marker, linestyle = backend_styles.get(
                    backend, (backend, "#333333", "o", "-")
                )
                
                ax.semilogy(xs, ys, label=label, marker=marker, color=color, 
                            linestyle=linestyle, linewidth=2.0, markersize=5)
                
            ax.set_xlabel("Number of Qubits", fontsize=11, labelpad=8)
            ax.set_ylabel("Execution Time (ms, log scale)", fontsize=11, labelpad=8)
            ax.set_title(f"Scaling Comparison: {bench.replace('_', ' ').title()} Workload", 
                         fontsize=13, pad=15, fontweight="bold")
            
            ax.grid(True, which="both", linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            # Position legend neatly on the right
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=9.5)
            plt.tight_layout()
            
            plot_file = plots_dir / f"{bench}_comparison.png"
            fig.savefig(plot_file, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {plot_file}")
            
    except Exception as e:
        print(f"Plotting failed: {e}")

    print("--- Benchmark Orchestration Done ---")

if __name__ == "__main__":
    main()
