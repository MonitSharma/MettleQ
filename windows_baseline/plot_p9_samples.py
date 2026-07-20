import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    run_dir = Path("windows_baseline/p9_submission")
    summary = json.loads((run_dir / "summary.json").read_text())
    
    counts = summary.get("counts") or {}
    if not counts:
        print("No counts dictionary found in summary.json")
        return
        
    # Sort counts by value descending
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_n = 10
    top = sorted_counts[:top_n]
    top.reverse()  # largest at the top of the horizontal bar chart
    
    bitstrings = [bs for bs, _ in top]
    counts_list = [c for _, c in top]
    
    predicted = summary.get("predicted_bitstring", "")
    
    PEAK_COLOR = "#c0392b"
    BAR_COLOR = "#5b86b5"
    colors = [PEAK_COLOR if bs == predicted else BAR_COLOR for bs in bitstrings]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(bitstrings, counts_list, color=colors)
    
    # Customize grid, labels, spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    
    ax.set_xlabel("Count (out of 1000 shots)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title("56-qubit P9 Peaked Circuit Top 10 Bitstring Samples (Predicted Peak in Red)", fontsize=13, fontweight='bold', pad=15)
    
    # Annotate counts on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                va='center', ha='left', fontsize=10, fontweight='bold')
                
    plt.tight_layout()
    
    plots_dir = Path("windows_baseline/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "p9_samples.png", dpi=300)
    print("Successfully generated windows_baseline/plots/p9_samples.png")

if __name__ == "__main__":
    main()
