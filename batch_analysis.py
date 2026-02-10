"""
Batch Analysis Script
Analyze multiple subjects and generate comparative reports
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main_pipeline import EMGNetworkPipeline
from utils import load_config


def run_batch_analysis(subject_ids, output_dir='outputs/batch_analysis'):
    """
    Run batch analysis on multiple subjects.
    
    Parameters
    ----------
    subject_ids : list
        List of subject IDs to analyze
    output_dir : str
        Directory for output files
    """
    # Initialize pipeline
    pipeline = EMGNetworkPipeline()
    
    # Analyze all subjects
    print(f"\n{'='*80}")
    print(f"BATCH ANALYSIS: {len(subject_ids)} subjects")
    print(f"{'='*80}\n")
    
    results_df = pipeline.analyze_multiple_subjects(subject_ids)
    
    # Generate comprehensive report
    pipeline.generate_report(results_df)
    
    # Additional statistical analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80 + "\n")
    
    # Compute correlations between metrics
    metric_cols = [col for col in results_df.columns if col.startswith('network_')]
    
    if len(metric_cols) > 1:
        corr_matrix = results_df[metric_cols].corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1
        )
        plt.title('Network Metrics Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        corr_path = Path(output_dir) / 'metric_correlations.png'
        corr_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation matrix to {corr_path}")
        plt.close()
    
    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics_to_plot = [
        'network_density',
        'network_global_efficiency',
        'network_avg_clustering',
        'network_modularity',
        'network_assortativity',
        'network_small_worldness'
    ]
    
    for idx, metric in enumerate(metrics_to_plot):
        if idx >= len(axes) or metric not in results_df.columns:
            continue
        
        axes[idx].hist(results_df[metric], bins=15, alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel(metric.replace('network_', '').replace('_', ' ').title())
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'Distribution: {metric.replace("network_", "").replace("_", " ").title()}')
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Network Metrics Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    dist_path = Path(output_dir) / 'metric_distributions.png'
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"Saved distributions to {dist_path}")
    plt.close()
    
    print("\nBatch analysis complete!")
    print(f"Results saved to: {pipeline.output_dir}")
    
    return results_df


def compare_groups(group1_ids, group2_ids, group_names=['Healthy', 'Pathological']):
    """
    Compare network metrics between two groups.
    
    Parameters
    ----------
    group1_ids : list
        Subject IDs for group 1
    group2_ids : list
        Subject IDs for group 2
    group_names : list
        Names for the groups
    """
    from scipy import stats
    
    # Analyze both groups
    pipeline = EMGNetworkPipeline()
    
    print(f"\nAnalyzing {group_names[0]} group ({len(group1_ids)} subjects)...")
    df1 = pipeline.analyze_multiple_subjects(group1_ids)
    df1['group'] = group_names[0]
    
    print(f"\nAnalyzing {group_names[1]} group ({len(group2_ids)} subjects)...")
    df2 = pipeline.analyze_multiple_subjects(group2_ids)
    df2['group'] = group_names[1]
    
    # Combine dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Statistical comparison
    metric_cols = [col for col in combined_df.columns if col.startswith('network_')]
    
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80 + "\n")
    
    comparison_results = []
    
    for metric in metric_cols:
        vals1 = df1[metric].dropna()
        vals2 = df2[metric].dropna()
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(vals1, vals2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(vals1)-1)*vals1.std()**2 + (len(vals2)-1)*vals2.std()**2) / (len(vals1)+len(vals2)-2))
        cohen_d = (vals1.mean() - vals2.mean()) / pooled_std
        
        comparison_results.append({
            'metric': metric,
            f'{group_names[0]}_mean': vals1.mean(),
            f'{group_names[0]}_std': vals1.std(),
            f'{group_names[1]}_mean': vals2.mean(),
            f'{group_names[1]}_std': vals2.std(),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohen_d': cohen_d,
            'significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    # Display results
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comp_path = pipeline.output_dir / 'group_comparison.csv'
    comparison_df.to_csv(comp_path, index=False)
    print(f"\nSaved comparison to {comp_path}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metric_cols[:6]):
        if idx >= len(axes):
            break
        
        # Box plot
        data_to_plot = [df1[metric].dropna(), df2[metric].dropna()]
        bp = axes[idx].boxplot(data_to_plot, labels=group_names, patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[idx].set_ylabel('Value')
        axes[idx].set_title(metric.replace('network_', '').replace('_', ' ').title())
        axes[idx].grid(alpha=0.3)
        
        # Add significance marker
        row = comparison_df[comparison_df['metric'] == metric].iloc[0]
        if row['p_value'] < 0.05:
            y_max = max(df1[metric].max(), df2[metric].max())
            axes[idx].text(1.5, y_max * 1.1, f"p={row['p_value']:.3f}*", 
                          ha='center', fontweight='bold')
    
    plt.suptitle(f'Group Comparison: {group_names[0]} vs {group_names[1]}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comp_viz_path = pipeline.output_dir / 'group_comparison.png'
    plt.savefig(comp_viz_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {comp_viz_path}")
    plt.close()
    
    return comparison_df, combined_df


if __name__ == '__main__':
    # Example: Batch analysis
    subject_ids = ['S2', 'S3', 'S4', 'S5']
    results = run_batch_analysis(subject_ids)
    
    # Example: Group comparison (uncomment if you have pathological data)
    # healthy_ids = ['S2', 'S3', 'S4']
    # pathological_ids = ['P1', 'P2', 'P3']
    # comparison = compare_groups(healthy_ids, pathological_ids)