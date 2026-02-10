"""
Create clean summary visualizations from compiled results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

def create_summary_plots(csv_path):
    """Create summary visualizations from compiled_results.csv"""
    
    # Read results
    results_df = pd.read_csv(csv_path)
    output_dir = Path(csv_path).parent / 'summary_plots'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Creating Summary Visualizations")
    print(f"{'='*80}\n")
    print(f"Subjects: {len(results_df)}")
    print(f"Output: {output_dir}\n")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    # Get metric columns
    metric_cols = [col for col in results_df.columns if col.startswith('network_')]
    
    # 1. Key Metrics Overview
    print("📊 Creating metrics overview...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    key_metrics = [
        'network_density',
        'network_global_efficiency',
        'network_avg_clustering',
        'network_modularity',
        'network_assortativity',
        'network_small_worldness'
    ]
    
    for idx, metric in enumerate(key_metrics):
        if metric in results_df.columns:
            ax = axes[idx]
            data = results_df[metric].dropna()
            
            # Histogram
            ax.hist(data, bins=12, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Add mean line
            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            
            # Labels
            metric_name = metric.replace('network_', '').replace('_', ' ').title()
            ax.set_xlabel(metric_name, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{metric_name}\n(n={len(data)})', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
    
    plt.suptitle('Network Metrics Distribution Across All Subjects', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / '1_metrics_distribution.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    # 2. Subject Comparison
    print("📊 Creating subject comparison...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Select key metrics for comparison
    comparison_metrics = ['network_density', 'network_global_efficiency', 
                         'network_modularity', 'network_avg_clustering']
    comparison_metrics = [m for m in comparison_metrics if m in results_df.columns]
    
    # Normalize for comparison
    df_plot = results_df[['subject_id'] + comparison_metrics].copy()
    for col in comparison_metrics:
        df_plot[col] = (df_plot[col] - df_plot[col].mean()) / df_plot[col].std()
    
    df_plot = df_plot.set_index('subject_id')
    
    # Heatmap
    sns.heatmap(df_plot.T, cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Z-score'}, ax=ax,
                linewidths=0.5, vmin=-2, vmax=2, annot=False)
    
    ax.set_title('Normalized Network Metrics by Subject', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Subject ID', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    
    # Simplify y-labels
    ylabels = [label.get_text().replace('network_', '').replace('_', ' ').title() 
               for label in ax.get_yticklabels()]
    ax.set_yticklabels(ylabels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_subject_comparison.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    # 3. Summary Statistics
    print("📊 Creating summary statistics...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate statistics
    stats_df = results_df[metric_cols].describe().T[['mean', 'std', 'min', 'max']]
    stats_df['CV'] = stats_df['std'] / stats_df['mean']
    stats_df = stats_df.round(3)
    
    # Simplify metric names
    stats_df.index = [idx.replace('network_', '').replace('_', ' ').title() 
                      for idx in stats_df.index]
    
    # Create table
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=stats_df.values,
                     rowLabels=stats_df.index,
                     colLabels=['Mean', 'Std', 'Min', 'Max', 'CV'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15] * 5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Style
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(stats_df) + 1):
        for j in range(-1, 5):
            table[(i, j)].set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    plt.title(f'Summary Statistics - Network Metrics (n={len(results_df)} subjects)',
             fontsize=13, fontweight='bold', pad=20)
    plt.savefig(output_dir / '3_summary_statistics.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    # 4. Box Plots
    print("📊 Creating box plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(key_metrics):
        if metric in results_df.columns and idx < len(axes):
            ax = axes[idx]
            data = results_df[metric].dropna()
            
            # Box plot
            bp = ax.boxplot([data], labels=[''], patch_artist=True, widths=0.6)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            
            # Add scatter points
            y = data.values
            x = np.random.normal(1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=40, color='navy')
            
            # Stats annotations
            ax.text(1.4, data.median(), f'Median: {data.median():.3f}', 
                   fontsize=9, va='center')
            
            metric_name = metric.replace('network_', '').replace('_', ' ').title()
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xlim(0.5, 1.5)
    
    plt.suptitle('Distribution of Network Metrics', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / '4_boxplots.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    # 5. Correlation Matrix
    print("📊 Creating correlation matrix...")
    if len(metric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr = results_df[metric_cols].corr()
        
        # Mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={'label': 'Correlation'},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Correlation Between Network Metrics', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Simplify labels
        labels = [label.replace('network_', '').replace('_', ' ').title() 
                 for label in corr.columns]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_dir / '5_correlation_matrix.png', bbox_inches='tight', dpi=200)
        plt.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"✅ Summary visualizations created!")
    print(f"{'='*80}")
    print(f"\nKey Findings:")
    print(f"  Subjects analyzed: {len(results_df)}")
    if 'n_cycles' in results_df.columns:
        print(f"  Avg gait cycles: {results_df['n_cycles'].mean():.1f} ± {results_df['n_cycles'].std():.1f}")
    if 'network_density' in results_df.columns:
        print(f"  Network density: {results_df['network_density'].mean():.3f} ± {results_df['network_density'].std():.3f}")
    if 'network_global_efficiency' in results_df.columns:
        print(f"  Global efficiency: {results_df['network_global_efficiency'].mean():.3f} ± {results_df['network_global_efficiency'].std():.3f}")
    if 'network_modularity' in results_df.columns:
        print(f"  Modularity: {results_df['network_modularity'].mean():.3f} ± {results_df['network_modularity'].std():.3f}")
    
    print(f"\n📁 Plots saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    # Find most recent results
    import glob
    
    csv_files = glob.glob('outputs/*/compiled_results.csv')
    if not csv_files:
        print("❌ No compiled_results.csv found!")
        print("Run main_pipeline.py first to analyze all subjects.")
        sys.exit(1)
    
    # Use most recent
    latest = max(csv_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"Using: {latest}")
    
    create_summary_plots(latest)