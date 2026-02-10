"""
Run analysis for all subjects and generate summary visualizations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main_pipeline import EMGNetworkPipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_all_subjects():
    """Analyze all subjects and create summary visualizations."""
    
    # Initialize pipeline
    pipeline = EMGNetworkPipeline()
    
    # Get all subject IDs
    from utils import get_subject_ids
    subject_ids = get_subject_ids(pipeline.config['dataset']['data_dir'])
    
    print(f"\n{'='*80}")
    print(f"Found {len(subject_ids)} subjects: {subject_ids}")
    print(f"{'='*80}\n")
    
    # Analyze all subjects
    print("Starting batch analysis...")
    results_df = pipeline.analyze_multiple_subjects(subject_ids)
    
    # Generate summary report
    pipeline.generate_report(results_df)
    
    # Create clean visualizations
    print("\nCreating summary visualizations...")
    create_summary_plots(results_df, pipeline.output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis complete!")
    print(f"Results saved to: {pipeline.output_dir}")
    print(f"{'='*80}\n")
    
    return results_df


def create_summary_plots(results_df, output_dir):
    """Create clean summary visualizations."""
    
    viz_dir = output_dir / 'summary_visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    # 1. Network Metrics Distribution
    print("  - Creating metrics distribution plot...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = [
        'network_density',
        'network_global_efficiency',
        'network_avg_clustering',
        'network_modularity',
        'network_assortativity',
        'network_small_worldness'
    ]
    
    for idx, metric in enumerate(metrics):
        if metric in results_df.columns:
            ax = axes[idx]
            ax.hist(results_df[metric].dropna(), bins=15, 
                   alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(results_df[metric].mean(), color='red', 
                      linestyle='--', linewidth=2, label='Mean')
            ax.set_xlabel(metric.replace('network_', '').replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric.replace("network_", "").replace("_", " ").title()}')
            ax.legend()
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'metrics_distribution.png', bbox_inches='tight')
    plt.close()
    
    # 2. Subject Comparison Heatmap
    print("  - Creating subject comparison heatmap...")
    metric_cols = [col for col in results_df.columns if col.startswith('network_')]
    if len(metric_cols) > 0:
        # Normalize metrics for comparison
        df_normalized = results_df[['subject_id'] + metric_cols].set_index('subject_id')
        df_normalized = (df_normalized - df_normalized.mean()) / df_normalized.std()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(df_normalized.T, cmap='RdYlGn', center=0, 
                   cbar_kws={'label': 'Z-score'}, ax=ax,
                   linewidths=0.5, annot=False)
        ax.set_title('Network Metrics Across All Subjects (Normalized)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Subject ID', fontsize=12)
        ax.set_ylabel('Network Metric', fontsize=12)
        plt.tight_layout()
        plt.savefig(viz_dir / 'subjects_heatmap.png', bbox_inches='tight')
        plt.close()
    
    # 3. Box Plot Comparison
    print("  - Creating box plot comparison...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric in results_df.columns and idx < len(axes):
            ax = axes[idx]
            data = results_df[metric].dropna()
            
            bp = ax.boxplot([data], labels=['All Subjects'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            
            # Add individual points
            y = data.values
            x = np.random.normal(1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.5, s=50, color='navy')
            
            ax.set_ylabel('Value')
            ax.set_title(metric.replace('network_', '').replace('_', ' ').title())
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'metrics_boxplot.png', bbox_inches='tight')
    plt.close()
    
    # 4. Summary Statistics Table
    print("  - Creating summary statistics table...")
    summary_stats = results_df[metric_cols].describe().T
    summary_stats['CV'] = summary_stats['std'] / summary_stats['mean']  # Coefficient of variation
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = summary_stats[['mean', 'std', 'min', 'max', 'CV']].round(3)
    table_data.index = [idx.replace('network_', '').replace('_', ' ').title() 
                       for idx in table_data.index]
    
    table = ax.table(cellText=table_data.values,
                    rowLabels=table_data.index,
                    colLabels=['Mean', 'Std Dev', 'Min', 'Max', 'CV'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15] * 5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data) + 1):
        table[(i, -1)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')
    
    plt.title('Summary Statistics - Network Metrics Across All Subjects',
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(viz_dir / 'summary_statistics.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    # 5. Correlation Matrix
    print("  - Creating correlation matrix...")
    if len(metric_cols) > 1:
        corr_matrix = results_df[metric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={'label': 'Correlation'},
                   ax=ax)
        ax.set_title('Correlation Between Network Metrics', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Simplify labels
        labels = [label.replace('network_', '').replace('_', ' ').title() 
                 for label in corr_matrix.columns]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'correlation_matrix.png', bbox_inches='tight')
        plt.close()
    
    print(f"\n✅ Summary visualizations saved to: {viz_dir}")


if __name__ == '__main__':
    results = analyze_all_subjects()
    
    # Display quick summary
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    print(f"\nSubjects analyzed: {len(results)}")
    print(f"Average gait cycles per subject: {results['n_cycles'].mean():.1f}")
    print(f"\nKey Metrics (Mean ± Std):")
    
    if 'network_density' in results.columns:
        print(f"  Density: {results['network_density'].mean():.3f} ± {results['network_density'].std():.3f}")
    if 'network_global_efficiency' in results.columns:
        print(f"  Global Efficiency: {results['network_global_efficiency'].mean():.3f} ± {results['network_global_efficiency'].std():.3f}")
    if 'network_modularity' in results.columns:
        print(f"  Modularity: {results['network_modularity'].mean():.3f} ± {results['network_modularity'].std():.3f}")
    
    print("\n" + "="*80)