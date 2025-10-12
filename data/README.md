# FirePrint Data Directory

## 📁 Directory Structure

```
data/
├── Bushfire_Boundaries_Historical_2024_V3.gdb/   # Original geodatabase (324K fires)
├── demo_processed_data/                          # Processed demo fingerprints
├── demo_similarity_search/                       # Search indices
├── demo_training_models/                         # Trained models
├── fire_feature_database/                        # Extracted features
├── processed_data/                               # Additional processed data
└── shared_functions/                             # Shared utilities
```

## 📊 Dataset Information

### Australian Bushfire Boundaries Historical Dataset 2024 V3

- **Source**: Australian Government
- **Records**: 324,741 fire polygons
- **Time Range**: 1898-2024
- **Format**: ESRI Geodatabase (.gdb)
- **Size**: ~2.5 GB

### Processed Data

- **Fingerprints**: 224×224×4 NumPy arrays (.npy)
- **Features**: CSV files with 20+ geometric/textural features
- **Models**: Trained Keras models (.keras, .h5)
- **Indices**: Similarity search indices (.pkl)

## 🚫 Git Ignore

Large data files are excluded from version control via `.gitignore`:
- Raw geodatabase files
- Processed fingerprints (*.npy)
- Model weights (*.keras, *.h5)
- Search indices (*.pkl)

## 📥 Getting the Data

The original dataset can be downloaded from:
- Australian Government data portal
- Contact: [dataset information]

## 💾 Storage Requirements

- **Full Dataset**: ~100 GB (after processing)
- **Sample (1K fires)**: ~500 MB
- **Demo Data**: Included in repository

## 🔒 Data License

The Australian Bushfire dataset is subject to its own licensing terms.
Please refer to the original data source for usage rights.

