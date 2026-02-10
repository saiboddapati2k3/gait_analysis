"""Feature extraction package for EMG network analysis."""

from .coherence import (
    CoherenceAnalyzer,
    ConnectivityAnalyzer,
    TemporalCoherenceAnalyzer
)

__all__ = [
    'CoherenceAnalyzer',
    'ConnectivityAnalyzer',
    'TemporalCoherenceAnalyzer'
]