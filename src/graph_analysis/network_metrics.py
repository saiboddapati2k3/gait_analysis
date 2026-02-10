"""
Graph construction and network analysis module
Implements multiple graph construction methods and extracts network metrics
"""

import numpy as np
import networkx as nx
from scipy.stats import zscore
import community as community_louvain
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class GraphConstructor:
    """Construct network graphs from connectivity matrices."""
    
    def __init__(self, config: dict):
        """
        Initialize graph constructor.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.muscle_names = config['muscles']['names']
        
    def construct_graph(self, connectivity_matrix: np.ndarray, 
                       threshold_method: str = 'adaptive',
                       threshold_value: Optional[float] = None) -> nx.Graph:
        """
        Construct network graph from connectivity matrix.
        
        Parameters
        ----------
        connectivity_matrix : np.ndarray
            Connectivity/coherence matrix [n_channels, n_channels]
        threshold_method : str
            Method for thresholding: 'fixed', 'adaptive', 'MST_plus'
        threshold_value : float, optional
            Fixed threshold value (only used if threshold_method='fixed')
        
        Returns
        -------
        G : nx.Graph
            Network graph
        """
        if threshold_method == 'fixed':
            return self._construct_fixed_threshold(connectivity_matrix, threshold_value)
        elif threshold_method == 'adaptive':
            return self._construct_adaptive_threshold(connectivity_matrix)
        elif threshold_method == 'MST_plus':
            return self._construct_mst_plus(connectivity_matrix)
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    def _construct_fixed_threshold(self, matrix: np.ndarray, 
                                   threshold: float) -> nx.Graph:
        """Construct graph with fixed threshold."""
        G = nx.Graph()
        
        # Add nodes
        for muscle in self.muscle_names:
            G.add_node(muscle)
        
        # Add edges above threshold
        n = len(self.muscle_names)
        for i in range(n):
            for j in range(i+1, n):
                if matrix[i, j] > threshold:
                    G.add_edge(
                        self.muscle_names[i],
                        self.muscle_names[j],
                        weight=matrix[i, j]
                    )
        
        logger.info(f"Constructed graph with {G.number_of_edges()} edges "
                   f"(threshold={threshold:.3f})")
        return G
    
    def _construct_adaptive_threshold(self, matrix: np.ndarray) -> nx.Graph:
        """Construct graph with adaptive percentile-based threshold."""
        # Get upper triangle values (excluding diagonal)
        upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
        
        # Compute threshold as percentile
        percentile = self.config['coherence']['percentile_threshold']
        threshold = np.percentile(upper_tri, percentile)
        
        return self._construct_fixed_threshold(matrix, threshold)
    
    def _construct_mst_plus(self, matrix: np.ndarray) -> nx.Graph:
        """
        Construct graph using Minimum Spanning Tree plus strongest edges.
        This ensures connectivity while preserving important connections.
        """
        # Create complete weighted graph
        G_complete = nx.Graph()
        n = len(self.muscle_names)
        
        for i in range(n):
            for j in range(i+1, n):
                G_complete.add_edge(
                    self.muscle_names[i],
                    self.muscle_names[j],
                    weight=1.0 - matrix[i, j]  # Invert for MST
                )
        
        # Compute MST
        mst = nx.minimum_spanning_tree(G_complete)
        
        # Add strongest edges not in MST
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not mst.has_edge(self.muscle_names[i], self.muscle_names[j]):
                    edges.append((i, j, matrix[i, j]))
        
        # Sort by strength and add top edges
        edges.sort(key=lambda x: x[2], reverse=True)
        n_additional = min(n, len(edges))  # Add up to n additional edges
        
        for i, j, weight in edges[:n_additional]:
            mst.add_edge(self.muscle_names[i], self.muscle_names[j], weight=weight)
        
        # Restore original weights
        for u, v in mst.edges():
            i = self.muscle_names.index(u)
            j = self.muscle_names.index(v)
            mst[u][v]['weight'] = matrix[i, j]
        
        logger.info(f"Constructed MST+ graph with {mst.number_of_edges()} edges")
        return mst
    
    def construct_directed_graph(self, causality_matrix: np.ndarray,
                                p_values: np.ndarray,
                                alpha: float = 0.05) -> nx.DiGraph:
        """
        Construct directed graph from Granger causality.
        
        Parameters
        ----------
        causality_matrix : np.ndarray
            Granger causality matrix
        p_values : np.ndarray
            P-values for significance testing
        alpha : float
            Significance level
        
        Returns
        -------
        G : nx.DiGraph
            Directed network graph
        """
        G = nx.DiGraph()
        
        # Add nodes
        for muscle in self.muscle_names:
            G.add_node(muscle)
        
        # Add significant directed edges
        n = len(self.muscle_names)
        for i in range(n):
            for j in range(n):
                if i != j and p_values[i, j] < alpha:
                    G.add_edge(
                        self.muscle_names[i],
                        self.muscle_names[j],
                        weight=causality_matrix[i, j],
                        p_value=p_values[i, j]
                    )
        
        logger.info(f"Constructed directed graph with {G.number_of_edges()} edges "
                   f"(alpha={alpha})")
        return G


class NetworkAnalyzer:
    """Extract network metrics from graphs."""
    
    def __init__(self, config: dict):
        """Initialize network analyzer."""
        self.config = config
        
    def extract_global_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """
        Extract global network metrics.
        
        Parameters
        ----------
        G : nx.Graph
            Network graph
        
        Returns
        -------
        metrics : dict
            Global network metrics
        """
        metrics = {}
        
        # Basic properties
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        
        # Check connectivity
        is_connected = nx.is_connected(G)
        metrics['is_connected'] = is_connected
        
        if is_connected:
            # Metrics requiring connected graph
            metrics['diameter'] = nx.diameter(G)
            metrics['average_path_length'] = nx.average_shortest_path_length(G)
            metrics['global_efficiency'] = nx.global_efficiency(G)
        else:
            # Use largest component
            largest_cc = max(nx.connected_components(G), key=len)
            G_connected = G.subgraph(largest_cc).copy()
            
            metrics['diameter'] = nx.diameter(G_connected)
            metrics['average_path_length'] = nx.average_shortest_path_length(G_connected)
            metrics['global_efficiency'] = nx.global_efficiency(G_connected)
            metrics['largest_component_size'] = len(largest_cc)
        
        # Clustering
        metrics['avg_clustering'] = nx.average_clustering(G)
        metrics['transitivity'] = nx.transitivity(G)
        
        # Assortativity
        try:
            metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            metrics['assortativity'] = 0.0
        
        # Small-worldness
        metrics['small_worldness'] = self._compute_small_worldness(G)
        
        # Modularity
        metrics['modularity'] = self._compute_modularity(G)
        
        logger.info(f"Extracted {len(metrics)} global metrics")
        return metrics
    
    def extract_nodal_metrics(self, G: nx.Graph) -> Dict[str, Dict[str, float]]:
        """
        Extract node-level metrics.
        
        Parameters
        ----------
        G : nx.Graph
            Network graph
        
        Returns
        -------
        nodal_metrics : dict
            Dictionary mapping node names to their metrics
        """
        nodal_metrics = {}
        
        # Centrality measures
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        closeness_cent = nx.closeness_centrality(G)
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
        pagerank = nx.pagerank(G)
        
        # Clustering coefficient
        clustering = nx.clustering(G)
        
        # Combine metrics for each node
        for node in G.nodes():
            nodal_metrics[node] = {
                'degree': G.degree(node),
                'degree_centrality': degree_cent[node],
                'betweenness_centrality': betweenness_cent[node],
                'closeness_centrality': closeness_cent[node],
                'eigenvector_centrality': eigenvector_cent[node],
                'pagerank': pagerank[node],
                'clustering': clustering[node]
            }
        
        return nodal_metrics
    
    def detect_communities(self, G: nx.Graph, 
                          algorithm: str = 'louvain') -> Dict[str, int]:
        """
        Detect communities in the network.
        
        Parameters
        ----------
        G : nx.Graph
            Network graph
        algorithm : str
            Community detection algorithm
        
        Returns
        -------
        communities : dict
            Mapping of nodes to community IDs
        """
        if algorithm == 'louvain':
            communities = community_louvain.best_partition(G)
        elif algorithm == 'label_propagation':
            communities_gen = nx.algorithms.community.label_propagation_communities(G)
            communities = {}
            for i, comm in enumerate(communities_gen):
                for node in comm:
                    communities[node] = i
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        n_communities = len(set(communities.values()))
        logger.info(f"Detected {n_communities} communities using {algorithm}")
        
        return communities
    
    def _compute_small_worldness(self, G: nx.Graph) -> float:
        """
        Compute small-world coefficient.
        Small-worldness = (C/C_rand) / (L/L_rand)
        where C is clustering, L is path length
        """
        if not nx.is_connected(G):
            return 0.0
        
        # Actual network metrics
        C = nx.average_clustering(G)
        L = nx.average_shortest_path_length(G)
        
        # Generate random network with same degree sequence
        try:
            degree_sequence = [d for n, d in G.degree()]
            G_rand = nx.configuration_model(degree_sequence)
            G_rand = nx.Graph(G_rand)  # Remove multi-edges
            G_rand.remove_edges_from(nx.selfloop_edges(G_rand))  # Remove self-loops
            
            if nx.is_connected(G_rand):
                C_rand = nx.average_clustering(G_rand)
                L_rand = nx.average_shortest_path_length(G_rand)
                
                sigma = (C / C_rand) / (L / L_rand)
                return sigma
        except:
            pass
        
        return 0.0
    
    def _compute_modularity(self, G: nx.Graph) -> float:
        """Compute modularity using Louvain algorithm."""
        communities = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(communities, G)
        return modularity
    
    def compare_networks(self, G1: nx.Graph, G2: nx.Graph) -> Dict[str, float]:
        """
        Compare two networks and compute difference metrics.
        
        Parameters
        ----------
        G1, G2 : nx.Graph
            Networks to compare
        
        Returns
        -------
        comparison : dict
            Comparison metrics
        """
        metrics1 = self.extract_global_metrics(G1)
        metrics2 = self.extract_global_metrics(G2)
        
        comparison = {}
        for key in metrics1.keys():
            if isinstance(metrics1[key], (int, float)) and isinstance(metrics2[key], (int, float)):
                comparison[f'{key}_diff'] = metrics2[key] - metrics1[key]
                if metrics1[key] != 0:
                    comparison[f'{key}_ratio'] = metrics2[key] / metrics1[key]
        
        return comparison