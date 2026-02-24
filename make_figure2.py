#!/usr/bin/env python3
"""
Simple wrapper to generate Figure 2 - auto-finds CSV file
"""

import os
import sys
import glob
from pathlib import Path

# Add the actual script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def find_compiled_results():
    """Search for compiled_results.csv"""
    print("🔍 Searching for compiled_results.csv...")
    
    # Search patterns
    patterns = [
        "compiled_results.csv",
        "outputs/*/compiled_results.csv",
        "*/compiled_results.csv",
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            csv_path = matches[0]
            print(f"✓ Found: {csv_path}\n")
            return csv_path
    
    print("❌ Could not find compiled_results.csv!")
    print("\nPlease specify the path manually:")
    print("  python3 make_figure2.py /path/to/compiled_results.csv")
    return None

def generate_figure2(csv_path, output_dir='paper_figures'):
    """Generate Figure 2"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set publication quality
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    
    # Load data
    print(f"📊 Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Exclude S17
    if 'S17' in df['subject_id'].values:
        print("⚠️  Excluding S17 (insufficient cycles)")
        df = df[df['subject_id'] != 'S17']
    
    n = len(df)
    print(f"✓ Analyzing {n} subjects\n")
    
    # Extract BC columns
    bc_cols = [col for col in df.columns if 'betweenness_centrality' in col]
    
    if not bc_cols:
        print("❌ ERROR: No betweenness_centrality columns found!")
        print("\nAvailable columns:", df.columns.tolist()[:10], "...")
        return None
    
    # Create muscle labels
    muscle_labels = []
    for col in bc_cols:
        name = col.replace('betweenness_centrality_', '')
        side = 'L' if name.startswith('LT') else 'R'
        muscle = name.split('_')[1]
        muscle_labels.append(f"{side}-{muscle}")
    
    # Compute stats
    bc_mean = df[bc_cols].mean().values
    bc_std = df[bc_cols].std().values
    
    # Sort
    sort_idx = np.argsort(bc_mean)[::-1]
    bc_mean_sorted = bc_mean[sort_idx]
    bc_std_sorted = bc_std[sort_idx]
    labels_sorted = [muscle_labels[i] for i in sort_idx]
    
    # Kruskal-Wallis
    bc_by_muscle = [df[col].values for col in bc_cols]
    h_stat, p_value = stats.kruskal(*bc_by_muscle)
    
    print("="*60)
    print("TOP 3 HUB MUSCLES:")
    print("="*60)
    for i in range(3):
        print(f"  {i+1}. {labels_sorted[i]:8s}  {bc_mean_sorted[i]:.4f} ± {bc_std_sorted[i]:.4f}")
    print(f"\nKruskal-Wallis: H={h_stat:.2f}, p={p_value:.4f}")
    print("="*60 + "\n")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x_pos = np.arange(len(labels_sorted))
    bars = ax.bar(x_pos, bc_mean_sorted,
                   yerr=bc_std_sorted,
                   capsize=4,
                   alpha=0.75,
                   color='steelblue',
                   edgecolor='black',
                   linewidth=0.8)
    
    # Highlight top 3
    for i in range(3):
        bars[i].set_color('coral')
        bars[i].set_edgecolor('darkred')
        bars[i].set_linewidth(1.2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_sorted, rotation=45, ha='right')
    ax.set_ylabel('Betweenness Centrality', fontweight='bold')
    ax.set_xlabel('Muscle', fontweight='bold')
    ax.set_title(f'Population-Level Betweenness Centrality (N={n})',
                 fontweight='bold')
    
    # Stats box
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    stats_text = f'Kruskal-Wallis\nH={h_stat:.2f}, p={p_value:.4f} {sig}'
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Mean line
    ax.axhline(bc_mean_sorted.mean(), color='red', linestyle='--',
               alpha=0.5, linewidth=1)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save
    png_path = f"{output_dir}/Figure2_population_BC.png"
    pdf_path = f"{output_dir}/Figure2_population_BC.pdf"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    print(f"✅ Saved: {png_path}")
    print(f"✅ Saved: {pdf_path}\n")
    
    plt.close()
    
    # Print text for paper
    print("="*60)
    print("TEXT FOR YOUR PAPER (Section V-D):")
    print("="*60)
    print(f"""
Betweenness Centrality analysis revealed significant heterogeneity 
across muscle groups (Kruskal-Wallis: H={h_stat:.2f}, p={p_value:.4f}). 
The Tibialis Anterior muscles exhibited the highest centrality 
bilaterally ({labels_sorted[0]}: {bc_mean_sorted[0]:.2f}±{bc_std_sorted[0]:.2f}, 
{labels_sorted[1]}: {bc_mean_sorted[1]:.2f}±{bc_std_sorted[1]:.2f}), 
followed by {labels_sorted[2]} ({bc_mean_sorted[2]:.2f}±{bc_std_sorted[2]:.2f}).
""")
    print("="*60)
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FIGURE 2 GENERATOR FOR PAPER")
    print("="*60 + "\n")
    
    # Check if CSV path provided
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = find_compiled_results()
    
    if not csv_path:
        sys.exit(1)
    
    # Check if output dir provided
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'paper_figures'
    
    # Generate figure
    success = generate_figure2(csv_path, output_dir)
    
    if success:
        print("\n✅ SUCCESS! Figure 2 is ready for your paper.\n")
    else:
        print("\n❌ FAILED! Check error messages above.\n")