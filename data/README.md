# Data Directory

This directory contains all data files for the FirePrint project. These files are **excluded from git** because they are too large.

## Directory Structure

```
data/
├── Bushfire_Boundaries_Historical_2024_V3.gdb/    # Source GDB database
├── demo_processed_data/                            # Demo processed fingerprints
│   ├── fingerprints.npy
│   ├── labels.pkl
│   ├── metadata.pkl
│   ├── encoders.json
│   └── processing_stats.json
├── demo_training_models/                           # Trained models
│   ├── best_model.keras
│   ├── demo_trained_model.keras
│   ├── demo_training_history.json
│   └── logs/
├── demo_similarity_search/                         # Search engines
│   ├── geometric_search.pkl
│   ├── cnn_search.pkl
│   ├── combined_search.pkl
│   └── *_scaler.pkl files
├── fire_feature_database/                          # Feature database
│   ├── raw_features.csv
│   ├── normalized_features.csv
│   ├── fire_metadata.csv
│   └── feature_metadata.json
└── processed_data/                                 # Production processed data
    └── (same structure as demo)
```

## File Descriptions

### Source Data
- **Bushfire_Boundaries_Historical_2024_V3.gdb**: ESRI Geodatabase with Australian bushfire polygons (1898-2024)

### Processed Data
- **fingerprints.npy**: Fire shape fingerprints (224x224 images)
- **labels.pkl**: Encoded labels for classification tasks
- **metadata.pkl**: Fire metadata (dates, locations, etc.)
- **encoders.json**: Label encoder mappings
- **processing_stats.json**: Processing statistics

### Models
- **best_model.keras**: Best performing model during training
- **demo_trained_model.keras**: Final trained model
- **demo_training_history.json**: Training history and metrics
- **logs/**: TensorBoard training logs

### Search Engines
- **geometric_search.pkl**: Similarity search based on geometric features
- **cnn_search.pkl**: Similarity search based on CNN features
- **combined_search.pkl**: Combined geometric + CNN search
- ***_scaler.pkl**: Feature scalers for normalization

### Feature Database
- **raw_features.csv**: Extracted geometric and texture features (unnormalized)
- **normalized_features.csv**: Normalized features for ML
- **fire_metadata.csv**: Metadata for each fire
- **feature_metadata.json**: Feature descriptions and statistics

## Configuration

All paths to these directories are managed in `FirePrint-v1.0/config.yaml`. To update paths:

```yaml
paths:
  data_root: "../data"
  source_data:
    bushfire_gdb: "../data/Bushfire_Boundaries_Historical_2024_V3.gdb"
  processed_data:
    demo: "../data/demo_processed_data"
    main: "../data/processed_data"
```

## Storage Requirements

- **GDB Database**: ~500 MB
- **Demo Processed Data**: ~200 MB
- **Models**: ~50 MB per model
- **Search Engines**: ~100 MB
- **Feature Database**: ~50 MB

**Total**: ~1-2 GB

## Git Ignore

These files are excluded from git via `.gitignore` to keep the repository size manageable. Only this README is tracked.

## Data Management

### Backup Recommendations
1. Keep original GDB file in a secure location
2. Back up trained models regularly
3. Version processed data with dates
4. Use cloud storage for large files

### Sharing Data
For sharing with collaborators:
1. Use cloud storage (Google Drive, Dropbox, etc.)
2. Share processed data, not raw GDB (unless needed)
3. Include the config.yaml so paths are consistent

### Regenerating Data
All data can be regenerated from the original GDB file using the notebooks:
1. `01_Fire_Polygon_to_Fingerprint.ipynb` - Create fingerprints
2. `02_Data_Processing_Pipeline.ipynb` - Process dataset
3. `03_CNN_Architecture_and_Training.ipynb` - Train models
4. `04_Pattern_Analysis_and_Features.ipynb` - Extract features
5. `05_Similarity_Search_and_Clustering.ipynb` - Build search engines
