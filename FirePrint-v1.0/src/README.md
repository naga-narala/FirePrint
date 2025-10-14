# FirePrint Source Code

## 📁 Directory Structure

This directory will contain the production-ready source code extracted from the Jupyter notebooks.

## 🚧 Current Status

The FirePrint v1.0 system is currently implemented in Jupyter notebooks (see `../notebooks/`). The source code modules are planned for future releases.

## 📋 Planned Modules

### Core Modules

- **`polygon_converter.py`**: Convert fire polygons to 4-channel fingerprints
  - `polygon_to_fingerprint()` - Main conversion function
  - `batch_convert()` - Batch processing
  - Channel generators (mask, distance, curvature, fractal)

- **`data_processor.py`**: Dataset processing pipeline
  - `FireDataProcessor` class
  - Data validation and cleaning
  - Label encoding and preparation
  - Train/test splitting

- **`cnn_model.py`**: CNN architecture definitions
  - `build_multi_task_cnn()` - Custom architecture
  - `build_transfer_learning_model()` - Transfer learning models
  - Loss functions and metrics

- **`feature_extractor.py`**: Feature extraction
  - `FirePatternAnalyzer` class
  - Shape features extraction
  - Texture features (GLCM)
  - Curvature analysis
  - Fractal dimension calculation

- **`similarity_search.py`**: Pattern matching engine
  - `FireSimilaritySearch` class
  - K-NN search implementation
  - Multi-modal feature fusion
  - Clustering utilities

- **`trainer.py`**: Training pipeline
  - `FireCNNTrainer` class
  - Training loop with callbacks
  - Evaluation metrics
  - Model saving/loading

### Utility Modules

- **`utils.py`**: Common utilities
  - Visualization helpers
  - File I/O functions
  - Configuration management

- **`config.py`**: Configuration settings
  - Model hyperparameters
  - File paths
  - Feature settings

## 🔨 Using the System

### Current Approach (v1.0)

Use the Jupyter notebooks directly:

```python
# See notebooks/01_Fire_Polygon_to_Fingerprint.ipynb for implementation
# All functions are defined within the notebooks
```

### Future Approach (v1.1+)

Once modules are implemented:

```python
from fireprint import polygon_to_fingerprint, FireCNNTrainer
from fireprint.similarity_search import FireSimilaritySearch

# Use as a regular Python package
```

## 🎯 Development Roadmap

1. **v1.1**: Extract core functions from notebooks to modules
2. **v1.2**: Add comprehensive unit tests
3. **v1.3**: Optimize performance and add caching
4. **v2.0**: Add REST API and web interface

## 🤝 Contributing

If you'd like to help convert notebooks to production code:

1. Pick a module from the planned modules list
2. Extract relevant code from notebooks
3. Add proper docstrings and type hints
4. Write unit tests
5. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

## 📖 Documentation

For now, refer to:
- **Notebooks**: `../notebooks/` - Complete implementation
- **Documentation**: `../docs/DOCUMENTATION.md` - Technical details
- **Examples**: `../examples/` - Usage examples

---

*This directory is a placeholder for future modular code development.*

