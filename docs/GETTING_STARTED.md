# 🚀 Getting Started with FirePrint v1.0

This guide will help you get up and running with FirePrint quickly.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.9 or higher** installed
- **NVIDIA GPU** with CUDA support (recommended, but not required)
- **16GB+ RAM** for processing large datasets
- **Git** installed for cloning the repository

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/FirePrint-v1.0.git
cd FirePrint-v1.0
```

### Step 2: Create Virtual Environment

**On Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import geopandas as gpd; print('GeoPandas installed successfully')"
```

## Quick Tour

### Option 1: Jupyter Notebooks (Recommended for Beginners)

FirePrint comes with 6 comprehensive Jupyter notebooks that walk you through the entire system:

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 01_Fire_Polygon_to_Fingerprint.ipynb
# 02_Data_Processing_Pipeline.ipynb
# 03_CNN_Architecture_and_Training.ipynb
# 04_Pattern_Analysis_and_Features.ipynb
# 05_Similarity_Search_and_Clustering.ipynb
# 06_Complete_System_Demo.ipynb
```

### Option 2: Python Scripts

Run the demo script:

```bash
cd examples
python fire_fingerprinting_cnn_demo.py
```

### Option 3: Interactive Python

```python
# Start Python interpreter
python

# Try basic fingerprint generation
from notebooks import *  # Load notebook functions
# Or implement your own using the examples
```

## Understanding the Workflow

FirePrint follows a clear pipeline:

```
1. Load Fire Data (GeoDataFrame)
   ↓
2. Convert to Fingerprints (4-channel images)
   ↓
3. Extract Features (geometric + textural)
   ↓
4. Train CNN Model (multi-task learning)
   ↓
5. Search & Analyze (similarity search, clustering)
```

## Example: Basic Usage

### 1. Load Fire Data

```python
import geopandas as gpd

# Load from geodatabase
gdf = gpd.read_file("data/Bushfire_Boundaries_Historical_2024_V3.gdb")

# Explore
print(f"Total fires: {len(gdf)}")
print(gdf.head())
```

### 2. Generate Fingerprints

```python
from src.polygon_converter import polygon_to_fingerprint
import matplotlib.pyplot as plt

# Convert single fire
fire_polygon = gdf.geometry[0]
fingerprint = polygon_to_fingerprint(fire_polygon)

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
channel_names = ['Shape Mask', 'Distance Transform', 'Curvature', 'Fractal']

for i, (ax, name) in enumerate(zip(axes, channel_names)):
    ax.imshow(fingerprint[:, :, i], cmap='viridis')
    ax.set_title(name)
    ax.axis('off')
plt.show()
```

### 3. Process Multiple Fires

```python
from src.data_processor import process_fire_dataset

# Process sample dataset
fingerprints, labels, metadata = process_fire_dataset(
    gdf,
    sample_size=100,  # Start small
    batch_size=10
)

print(f"Generated {len(fingerprints)} fingerprints")
print(f"Shape: {fingerprints.shape}")
```

### 4. Extract Features

```python
from src.feature_extractor import FirePatternAnalyzer

analyzer = FirePatternAnalyzer()
features = analyzer.batch_extract_features(fingerprints)

print(f"Extracted {features.shape[1]} features per fire")
print(features.head())
```

### 5. Train Model

```python
from src.cnn_model import build_multi_task_cnn
from src.trainer import FireCNNTrainer

# Build model
model = build_multi_task_cnn(input_shape=(224, 224, 4))

# Train
trainer = FireCNNTrainer(model, task_names=['fire_type', 'cause', 'state', 'size'])
history = trainer.train(
    X_train, y_train,
    X_val, y_val,
    epochs=10,  # Start with fewer epochs
    batch_size=16
)
```

### 6. Search Similar Fires

```python
from src.similarity_search import FireSimilaritySearch

# Create search engine
search = FireSimilaritySearch(features, labels)

# Find similar fires
query_idx = 0
similar = search.find_similar_fires(query_idx, n_neighbors=5)

print(f"Top 5 fires similar to fire #{query_idx}:")
for idx, similarity in similar:
    print(f"  Fire #{idx}: {similarity:.3f} similarity")
```

## Common Issues & Solutions

### Issue 1: Out of Memory

**Problem**: RAM exhausted when processing large datasets

**Solution**:
```python
# Process in smaller batches
fingerprints, labels, metadata = process_fire_dataset(
    gdf,
    sample_size=1000,  # Reduce sample size
    batch_size=50      # Reduce batch size
)
```

### Issue 2: CUDA/GPU Not Available

**Problem**: TensorFlow not using GPU

**Solution**:
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install GPU support
pip install tensorflow[and-cuda]
```

### Issue 3: Missing Dependencies

**Problem**: Import errors for specific packages

**Solution**:
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Or install individual package
pip install <package-name>
```

### Issue 4: Shapefile/GDB Read Errors

**Problem**: Can't read geodatabase

**Solution**:
```bash
# Install GDAL
pip install GDAL

# Or use conda
conda install -c conda-forge gdal
```

## Next Steps

1. **Explore Notebooks**: Go through all 6 notebooks to understand the system
2. **Try Different Features**: Experiment with feature extraction parameters
3. **Train Custom Models**: Modify CNN architecture for your needs
4. **Analyze Your Data**: Apply FirePrint to your own fire datasets
5. **Contribute**: Share improvements via GitHub pull requests

## Getting Help

- **Documentation**: See [DOCUMENTATION.md](DOCUMENTATION.md)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/FirePrint-v1.0/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/yourusername/FirePrint-v1.0/discussions)
- **Email**: Contact us at your.email@example.com

## Resources

- **Research Papers**: Coming soon
- **Video Tutorials**: Coming soon
- **Use Cases**: See [examples/](../examples/)
- **API Reference**: See code docstrings in [src/](../src/)

---

Happy fire pattern analysis! 🔥

*If you find FirePrint useful, please consider starring the repository on GitHub!*

