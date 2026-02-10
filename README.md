cat > README.md << 'EOF'
# Graph-Based Modeling of Inter-Muscle Coordination Using Surface EMG

**Research Project - Healthcare Data Analytics**



##  Project Overview

Advanced network analysis of muscle coordination patterns during gait using multi-channel surface EMG signals. Implements graph-theoretic frameworks to differentiate normal gait patterns.

##  Key Features

- **Multi-channel EMG Processing**: 10 muscle channels (bilateral lower limb)
- **Frequency-Specific Analysis**: 5 frequency bands (delta to gamma)
- **Network Construction**: Adaptive thresholding, MST+ algorithms
- **Comprehensive Metrics**: 17+ network measures
- **Community Detection**: Louvain algorithm for muscle synergies
- **Batch Processing**: Automated analysis for 31 subjects

##  Architecture
```
├── config/              # Analysis parameters (YAML)
├── src/
│   ├── data_loader/     # WFDB data loading & gait extraction
│   ├── preprocessing/   # Signal filtering & normalization
│   ├── feature_extraction/  # Coherence & connectivity
│   ├── graph_analysis/  # Network construction & metrics
│   ├── visualization/   # Publication-quality plots
│   └── utils/           # Helper functions
├── main_pipeline.py     # Main analysis script
├── create_summary_plots.py  # Summary visualizations
└── outputs/             # Results & figures
```

##  Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download EMG data to data/ folder
# From: https://physionet.org/content/semg/1.0.1/

# Run analysis
python3 main_pipeline.py

# Generate summary plots
python3 create_summary_plots.py
```

##  Results

- **Subjects Analyzed**: 31 healthy adults
- **Gait Cycles**: 150-250 per subject (avg)
- **Network Density**: 0.16-0.25 (mean coherence)
- **Communities**: 3-5 muscle groups per subject

##  Technologies

- **Python 3.12**
- **Signal Processing**: SciPy, WFDB
- **Network Analysis**: NetworkX, python-louvain
- **Visualization**: Matplotlib, Seaborn
- **Statistical Analysis**: Statsmodels, Pingouin

##  Research Foundation

Based on 10+ peer-reviewed papers including:
- Boonstra et al. (2015) - Muscle networks methodology
- Laine & Valero-Cuevas (2017) - Intermuscular coherence
- Locoratolo et al. (2024) - Graph-theoretic gait analysis

##  Dataset

**Surface Electromyographic Signals During Long-Lasting Ground Walking**  
Di Nardo et al., PhysioNet 2024  
https://physionet.org/content/semg/1.0.1/

- 31 subjects (20-30 years)
- 10 muscles (5 per leg, bilateral)
- 2000 Hz sampling rate
- ~5 minutes continuous walking

##  Key Outputs

1. **compiled_results.csv** - All network metrics for 31 subjects
2. **Summary visualizations** - Distribution, correlation, comparison plots
3. **Individual reports** - Per-subject network graphs & metrics
4. **Analysis report** - Statistical summary


---

**Note**: Raw EMG data files (.dat, .hea) are not included due to size. Download from PhysioNet link above.
EOF