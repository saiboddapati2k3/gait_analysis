"""
Visualization module for EMG network analysis
Includes network graphs, coherence matrices, and metric comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NetworkVisualizer:
    """Visualize muscle coordination networks."""
    
    def __init__(self, config: dict):
        """Initialize visualizer with configuration."""
        self.config = config
        self.muscle_names = config['muscles']['names']
        
        # Set style
        plt.style.use(config['visualization']['figure']['style'])
        self.dpi = config['visualization']['figure']['dpi']
        self.format = config['visualization']['figure']['format']
        
    def plot_network(self, G: nx.Graph, 
                    title: str = "Muscle Coordination Network",
                    layout: str = None,
                    save_path: Optional[str] = None,
                    node_colors: Optional[Dict] = None,
                    highlight_communities: bool = False) -> plt.Figure:
        """
        Plot network graph with customizable layout.
        
        Parameters
        ----------
        G : nx.Graph
            Network graph
        title : str
            Plot title
        layout : str, optional
            Layout algorithm
        save_path : str, optional
            Path to save figure
        node_colors : dict, optional
            Custom node colors
        highlight_communities : bool
            Whether to color nodes by community
        
        Returns
        -------
        fig : matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
        
        # Choose layout
        if layout is None:
            layout = self.config['visualization']['layout']
        
        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Node colors
        if highlight_communities:
            from community import best_partition
            communities = best_partition(G)
            node_color = [communities[node] for node in G.nodes()]
            cmap = plt.cm.tab10
        elif node_colors is not None:
            node_color = [node_colors.get(node, 0.5) for node in G.nodes()]
            cmap = plt.cm.viridis
        else:
            # Color by hemisphere (left vs right)
            node_color = ['#FF6B6B' if 'LT' in node else '#4ECDC4' for node in G.nodes()]
            cmap = None
        
        # Node sizes based on degree
        node_sizes = [1000 + 500 * G.degree(node) for node in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_color,
            node_size=node_sizes,
            cmap=cmap,
            alpha=0.9,
            ax=ax
        )
        
        # Draw edges with thickness based on weight
        if nx.get_edge_attributes(G, 'weight'):
            weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
            edge_widths = 0.5 + 4 * (weights / weights.max())
            
            # Color edges by weight
            edge_colors = weights
            
            nx.draw_networkx_edges(
                G, pos,
                width=edge_widths,
                edge_color=edge_colors,
                edge_cmap=plt.cm.coolwarm,
                alpha=0.6,
                ax=ax
            )
        else:
            nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, format=self.format, bbox_inches='tight')
            logger.info(f"Saved network plot to {save_path}")
        
        return fig
    
    def plot_coherence_matrix(self, matrix: np.ndarray,
                             title: str = "Intermuscular Coherence Matrix",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot coherence matrix as heatmap.
        
        Parameters
        ----------
        matrix : np.ndarray
            Coherence matrix
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        fig : matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(
            matrix,
            xticklabels=self.muscle_names,
            yticklabels=self.muscle_names,
            cmap='viridis',
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Coherence'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, format=self.format, bbox_inches='tight')
            logger.info(f"Saved coherence matrix to {save_path}")
        
        return fig
    
    def plot_multiband_coherence(self, coherence_bands: Dict[str, np.ndarray],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot coherence matrices for multiple frequency bands.
        
        Parameters
        ----------
        coherence_bands : dict
            Dictionary of band names to coherence matrices
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        fig : matplotlib.Figure
        """
        n_bands = len(coherence_bands)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        axes = axes.flatten()
        
        for idx, (band, matrix) in enumerate(coherence_bands.items()):
            if idx >= len(axes):
                break
            
            sns.heatmap(
                matrix,
                xticklabels=self.muscle_names,
                yticklabels=self.muscle_names,
                cmap='viridis',
                vmin=0,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Coherence'},
                ax=axes[idx]
            )
            
            axes[idx].set_title(f'{band.capitalize()} Band', fontsize=12, fontweight='bold')
            axes[idx].tick_params(labelsize=8)
        
        # Hide unused subplots
        for idx in range(n_bands, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Frequency-Specific Coherence Patterns', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, format=self.format, bbox_inches='tight')
            logger.info(f"Saved multiband coherence to {save_path}")
        
        return fig
    
    def plot_nodal_metrics(self, nodal_metrics: Dict[str, Dict],
                          metric_name: str = 'degree_centrality',
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot nodal metric values as bar chart.
        
        Parameters
        ----------
        nodal_metrics : dict
            Dictionary of node names to their metrics
        metric_name : str
            Which metric to plot
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        fig : matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        nodes = list(nodal_metrics.keys())
        values = [nodal_metrics[node][metric_name] for node in nodes]
        
        # Color by hemisphere
        colors = ['#FF6B6B' if 'LT' in node else '#4ECDC4' for node in nodes]
        
        bars = ax.bar(range(len(nodes)), values, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xticks(range(len(nodes)))
        ax.set_xticklabels(nodes, rotation=45, ha='right')
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Nodal {metric_name.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, format=self.format, bbox_inches='tight')
            logger.info(f"Saved nodal metrics to {save_path}")
        
        return fig
    
    def plot_metric_comparison(self, metrics1: Dict, metrics2: Dict,
                              labels: List[str] = ['Group 1', 'Group 2'],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare network metrics between two groups.
        
        Parameters
        ----------
        metrics1, metrics2 : dict
            Global metrics dictionaries
        labels : list
            Labels for the two groups
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        fig : matplotlib.Figure
        """
        # Select comparable metrics
        comparable_keys = [k for k in metrics1.keys() 
                          if isinstance(metrics1[k], (int, float)) and k in metrics2]
        
        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.dpi)
        
        x = np.arange(len(comparable_keys))
        width = 0.35
        
        values1 = [metrics1[k] for k in comparable_keys]
        values2 = [metrics2[k] for k in comparable_keys]
        
        ax.bar(x - width/2, values1, width, label=labels[0], alpha=0.8)
        ax.bar(x + width/2, values2, width, label=labels[1], alpha=0.8)
        
        ax.set_xlabel('Network Metrics', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Network Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace('_', ' ').title() for k in comparable_keys], 
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, format=self.format, bbox_inches='tight')
            logger.info(f"Saved metric comparison to {save_path}")
        
        return fig
    
    def plot_temporal_coherence(self, temporal_data: Dict,
                               muscle_pair: Tuple[int, int] = (0, 5),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time-varying coherence for a muscle pair.
        
        Parameters
        ----------
        temporal_data : dict
            Output from temporal coherence analysis
        muscle_pair : tuple
            Indices of muscles to plot
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        fig : matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=(14, 5), dpi=self.dpi)
        
        time = temporal_data['time']
        coherence = temporal_data['coherence'][:, muscle_pair[0], muscle_pair[1]]
        
        ax.plot(time, coherence, linewidth=2, color='#2E86AB')
        ax.fill_between(time, 0, coherence, alpha=0.3, color='#2E86AB')
        
        muscle1 = self.muscle_names[muscle_pair[0]]
        muscle2 = self.muscle_names[muscle_pair[1]]
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Coherence', fontsize=12)
        ax.set_title(f'Temporal Coherence: {muscle1} - {muscle2}', 
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, format=self.format, bbox_inches='tight')
            logger.info(f"Saved temporal coherence to {save_path}")
        
        return fig