import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def get_top_n(summary_path, n=10):
    summary = json.loads(summary_path.read_text())
    counts = summary.get("counts") or {}
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top = sorted_counts[:n]
    top.reverse()  # largest at top
    bitstrings = [bs for bs, _ in top]
    counts_list = [c for _, c in top]
    predicted = summary.get("predicted_bitstring", "")
    return bitstrings, counts_list, predicted

def main():
    wsl_path = Path("windows_baseline/p9_submission/summary.json")
    mac_path = Path("bench/runs/midpoint-mpo-p9-optimized/summary.json")
    
    if not wsl_path.exists():
        print(f"Error: {wsl_path} does not exist.")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    
    # Plot WSL
    bitstrings_wsl, counts_wsl, pred_wsl = get_top_n(wsl_path)
    PEAK_COLOR = "#c0392b"
    BAR_COLOR = "#76B900" # Green for WSL
    colors_wsl = [PEAK_COLOR if bs == pred_wsl else BAR_COLOR for bs in bitstrings_wsl]
    labels_wsl = [bs[:8] + "..." + bs[-8:] for bs in bitstrings_wsl]
    bars_wsl = axes[0].barh(labels_wsl, counts_wsl, color=colors_wsl)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].xaxis.grid(True, linestyle="--", alpha=0.6)
    axes[0].set_axisbelow(True)
    axes[0].set_xlabel("Count (out of 1000 shots)", fontsize=12, fontweight='bold', labelpad=10)
    axes[0].set_title("Windows WSL (i9-12900K, MKL + P-cores)\nPeak Fraction: 5.6% (56 counts)", fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlim(0, max(counts_wsl) * 1.15)
    for bar in bars_wsl:
        width = bar.get_width()
        if width >= 5:
            axes[0].text(width + max(counts_wsl)*0.015, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                         va='center', ha='left', fontsize=10, fontweight='bold')
                     
    # Plot Mac
    if mac_path.exists():
        bitstrings_mac, counts_mac, pred_mac = get_top_n(mac_path)
        BAR_COLOR_MAC = "#167D8D" # Teal for Mac M3 Pro
        colors_mac = [PEAK_COLOR if bs == pred_mac else BAR_COLOR_MAC for bs in bitstrings_mac]
        labels_mac = [bs[:8] + "..." + bs[-8:] for bs in bitstrings_mac]
        bars_mac = axes[1].barh(labels_mac, counts_mac, color=colors_mac)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].xaxis.grid(True, linestyle="--", alpha=0.6)
        axes[1].set_axisbelow(True)
        axes[1].set_xlabel("Count (out of 1000 shots)", fontsize=12, fontweight='bold', labelpad=10)
        axes[1].set_title("Apple Mac CPU (M3 Pro, 12-core)\nPeak Fraction: 10.0% (100 counts)", fontsize=13, fontweight='bold', pad=10)
        axes[1].set_xlim(0, max(counts_mac) * 1.15)
        for bar in bars_mac:
            width = bar.get_width()
            if width >= 5:
                axes[1].text(width + max(counts_mac)*0.015, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                             va='center', ha='left', fontsize=10, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, "Mac CPU data not found", va='center', ha='center', fontsize=14)
        
    fig.suptitle("56-qubit P9 Peaked Circuit Top 10 Bitstring Samples Comparison\n(Predicted Peak in Red)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    
    plots_dir = Path("windows_baseline/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "p9_samples.png", dpi=300)
    print("Successfully generated windows_baseline/plots/p9_samples.png")

if __name__ == "__main__":
    main()
