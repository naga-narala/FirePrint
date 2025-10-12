# FirePrint Outputs Directory

## 📁 Contents

This directory contains generated outputs from FirePrint analysis:

- **Feature CSV files**: Extracted geometric and textural features
- **NumPy arrays**: Processed feature matrices
- **Pickle files**: Serialized Python objects (scalers, encoders)

## 📝 File Types

### CSV Files
- `demo_fire_features.csv` - Raw feature values
- `demo_fire_features_normalized.csv` - Normalized features

### NumPy Arrays (.npy)
- `demo_cnn_features.npy` - CNN-extracted feature vectors

### Metadata
- Feature statistics
- Processing logs
- Label encodings

## 🚫 Git Ignore

Output files are excluded from version control to keep repository size small.
Users generate their own outputs by running the notebooks.

## 🔄 Regenerating Outputs

To regenerate outputs:

```bash
# Run the notebooks in order
jupyter notebook notebooks/
```

Or use the Python scripts:

```bash
python examples/fire_fingerprinting_cnn_demo.py
```

