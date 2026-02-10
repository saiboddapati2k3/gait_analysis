"""
Main analysis pipeline for EMG Network Analysis
Graph-Based Modeling of Inter-Muscle Coordination
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import (
    load_config, setup_logging, create_output_directory,
    save_results, get_subject_ids, NumpyEncoder
)
from data_loader import EMGDataLoader
from preprocessing import EMGPreprocessor
from feature_extraction import CoherenceAnalyzer, ConnectivityAnalyzer, TemporalCoherenceAnalyzer
from graph_analysis import GraphConstructor, NetworkAnalyzer
from visualization import NetworkVisualizer


class EMGNetworkPipeline:
    """Complete analysis pipeline for EMG network analysis."""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize pipeline with configuration."""
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config)
        self.logger.info("=" * 80)
        self.logger.info("EMG Network Analysis Pipeline Initialized")
        self.logger.info("=" * 80)
        
        # Create output directory
        self.output_dir = create_output_directory(
            self.config['dataset']['output_dir']
        )
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Initialize components
        self.data_loader = EMGDataLoader(
            self.config['dataset']['data_dir'],
            self.config
        )
        self.preprocessor = EMGPreprocessor(self.config)
        self.coherence_analyzer = CoherenceAnalyzer(self.config)
        self.connectivity_analyzer = ConnectivityAnalyzer(self.config)
        self.temporal_analyzer = TemporalCoherenceAnalyzer(self.config)
        self.graph_constructor = GraphConstructor(self.config)
        self.network_analyzer = NetworkAnalyzer(self.config)
        self.visualizer = NetworkVisualizer(self.config)
        
        # Storage for results
        self.results = {}
        
    def analyze_subject(self, subject_id: str, analyze_temporal: bool = True) -> dict:
        """
        Run complete analysis for a single subject.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        analyze_temporal : bool
            Whether to perform temporal analysis
        
        Returns
        -------
        subject_results : dict
            All analysis results for the subject
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Analyzing Subject: {subject_id}")
        self.logger.info(f"{'='*80}\n")
        
        start_time = time.time()
        subject_results = {'subject_id': subject_id}
        
        try:
            # Step 1: Load data
            self.logger.info("Step 1/7: Loading data...")
            data = self.data_loader.load_subject(subject_id)
            subject_results['data_info'] = {
                'duration': data['duration'],
                'sampling_rate': data['fs']
            }
            
            # Step 2: Preprocess EMG
            self.logger.info("Step 2/7: Preprocessing EMG signals...")
            emg_processed = self.preprocessor.preprocess(data['emg'])
            
            # Compute quality metrics
            quality_metrics = self.preprocessor.compute_quality_metrics(emg_processed)
            subject_results['quality_metrics'] = quality_metrics
            
            # Step 3: Extract gait cycles
            self.logger.info("Step 3/7: Extracting gait cycles...")
            events = self.data_loader.detect_gait_events(data['footswitch'], data['fs'])
            cycles = self.data_loader.extract_gait_cycles(
                emg_processed, events, data['fs'], foot='left'
            )
            
            if len(cycles) > 0:
                normalized_cycles = self.data_loader.normalize_cycles(cycles)
                subject_results['n_cycles'] = len(cycles)
            else:
                subject_results['n_cycles'] = 0

            # Use full preprocessed signal for analysis (better frequency resolution)
            mean_cycle = emg_processed
            
            # Step 4: Coherence analysis
            self.logger.info("Step 4/7: Computing intermuscular coherence...")
            coherence_bands = self.coherence_analyzer.compute_multiband_coherence(mean_cycle)
            subject_results['coherence'] = {
                band: matrix.tolist() for band, matrix in coherence_bands.items()
            }
            
            # Step 5: Connectivity analysis
            self.logger.info("Step 5/7: Computing connectivity measures...")
            correlation_matrix = self.connectivity_analyzer.compute_correlation(mean_cycle)
            mi_matrix = self.connectivity_analyzer.compute_mutual_information(mean_cycle)
            
            subject_results['correlation'] = correlation_matrix.tolist()
            subject_results['mutual_information'] = mi_matrix.tolist()
            
            # Granger causality (optional, can be slow)
            if self.config['advanced_features']['granger_causality']['enabled']:
                self.logger.info("Computing Granger causality...")
                gc_matrix, gc_pvalues = self.connectivity_analyzer.compute_granger_causality(
                    mean_cycle,
                    max_lag=self.config['advanced_features']['granger_causality']['max_lag']
                )
                subject_results['granger_causality'] = gc_matrix.tolist()
                subject_results['gc_pvalues'] = gc_pvalues.tolist()
            
            # Step 6: Graph construction and analysis
            self.logger.info("Step 6/7: Constructing and analyzing networks...")
            
            # Construct graph from coherence
            threshold_method = self.config['graph']['threshold_method']
            G = self.graph_constructor.construct_graph(
                coherence_bands['full'],
                threshold_method=threshold_method
            )
            
            # Extract network metrics
            global_metrics = self.network_analyzer.extract_global_metrics(G)
            nodal_metrics = self.network_analyzer.extract_nodal_metrics(G)
            communities = self.network_analyzer.detect_communities(G)
            
            subject_results['network'] = {
                'global_metrics': global_metrics,
                'nodal_metrics': nodal_metrics,
                'communities': communities
            }
            
            # Step 7: Temporal analysis (optional)
            if analyze_temporal and self.config['advanced_features']['temporal']['enabled']:
                self.logger.info("Step 7/7: Performing temporal analysis...")
                temporal_coherence = self.temporal_analyzer.compute_sliding_window_coherence(
                    emg_processed,
                    window_size=self.config['advanced_features']['temporal']['window_size'],
                    overlap=self.config['advanced_features']['temporal']['overlap']
                )
                coherence_variability = self.temporal_analyzer.compute_coherence_variability(
                    temporal_coherence
                )
                subject_results['temporal'] = {
                    'variability': coherence_variability.tolist()
                }
            
            # Visualization
            self.logger.info("Generating visualizations...")
            viz_dir = self.output_dir / 'figures' / subject_id
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Network graph
            self.visualizer.plot_network(
                G,
                title=f"Muscle Network - {subject_id}",
                save_path=viz_dir / f'{subject_id}_network.{self.visualizer.format}',
                highlight_communities=True
            )
            
            # Coherence matrix
            self.visualizer.plot_coherence_matrix(
                coherence_bands['full'],
                title=f"Coherence Matrix - {subject_id}",
                save_path=viz_dir / f'{subject_id}_coherence.{self.visualizer.format}'
            )
            
            # Multiband coherence
            self.visualizer.plot_multiband_coherence(
                coherence_bands,
                save_path=viz_dir / f'{subject_id}_multiband.{self.visualizer.format}'
            )
            
            # Nodal metrics
            self.visualizer.plot_nodal_metrics(
                nodal_metrics,
                metric_name='betweenness_centrality',
                save_path=viz_dir / f'{subject_id}_centrality.{self.visualizer.format}'
            )
            
            elapsed_time = time.time() - start_time
            subject_results['analysis_time'] = elapsed_time
            self.logger.info(f"Subject {subject_id} analyzed in {elapsed_time:.2f}s")
            
            # Save subject results
            results_path = self.output_dir / 'features' / f'{subject_id}_results.json'
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(subject_results, f, indent=2, cls=NumpyEncoder)
            
            return subject_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing subject {subject_id}: {str(e)}")
            raise
    
    def analyze_multiple_subjects(self, subject_ids: list = None) -> pd.DataFrame:
        """
        Analyze multiple subjects and compile results.
        
        Parameters
        ----------
        subject_ids : list, optional
            List of subject IDs. If None, analyze all available subjects.
        
        Returns
        -------
        results_df : pd.DataFrame
            Compiled results for all subjects
        """
        if subject_ids is None:
            subject_ids = get_subject_ids(self.config['dataset']['data_dir'])
        
        self.logger.info(f"Analyzing {len(subject_ids)} subjects...")
        
        all_results = []
        
        for subject_id in subject_ids:
            try:
                results = self.analyze_subject(subject_id, analyze_temporal=False)
                
                # Flatten results for DataFrame
                flat_results = {
                    'subject_id': subject_id,
                    'n_cycles': results.get('n_cycles', 0),
                    'duration': results['data_info']['duration']
                }
                
                # Add global network metrics
                for key, value in results['network']['global_metrics'].items():
                    flat_results[f'network_{key}'] = value
                
                all_results.append(flat_results)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {subject_id}: {str(e)}")
                continue
        
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save compiled results
        csv_path = self.output_dir / 'compiled_results.csv'
        results_df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved compiled results to {csv_path}")
        
        return results_df
    
    def generate_report(self, results_df: pd.DataFrame):
        """Generate summary report."""
        report_path = self.output_dir / 'reports' / 'analysis_summary.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EMG NETWORK ANALYSIS - SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Number of subjects analyzed: {len(results_df)}\n\n")
            
            f.write("Network Metrics Summary:\n")
            f.write("-" * 40 + "\n")
            
            metric_cols = [col for col in results_df.columns if col.startswith('network_')]
            summary_stats = results_df[metric_cols].describe()
            f.write(summary_stats.to_string())
            f.write("\n\n")
        
        self.logger.info(f"Generated report: {report_path}")


def main():
    """Main execution function."""
    # Initialize pipeline
    pipeline = EMGNetworkPipeline()
    
    # Get all subject IDs
    from utils import get_subject_ids
    subject_ids = get_subject_ids(pipeline.config['dataset']['data_dir'])
    
    print(f"\n{'='*80}")
    print(f"🚀 Analyzing {len(subject_ids)} subjects")
    print(f"{'='*80}\n")
    
    # Analyze all subjects
    results_df = pipeline.analyze_multiple_subjects(subject_ids)
    
    # Generate report
    pipeline.generate_report(results_df)
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis complete!")
    print(f"Results: {pipeline.output_dir}/compiled_results.csv")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()