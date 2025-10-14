# Migration Guide: Using config.yaml in Notebooks

## 🎯 What Changed

All file paths are now centralized in `config.yaml`. Instead of hardcoding paths in each notebook, you load them from the configuration file.

## 📝 Quick Migration for Each Notebook

### Notebook 01: Fire Polygon to Fingerprint

**Add at the top (after imports):**
```python
from src.config_loader import FirePrintConfig
config = FirePrintConfig()
```

**Replace:**
```python
# OLD
gdb_path = "../Forest_Fires/Bushfire_Boundaries_Historical_2024_V3.gdb"
```

**With:**
```python
# NEW
gdb_path = str(config.get_path('source_data.bushfire_gdb'))
```

---

### Notebook 02: Data Processing Pipeline

**Add at the top:**
```python
from src.config_loader import FirePrintConfig
config = FirePrintConfig()
```

**Replace:**
```python
# OLD
def __init__(self, gdb_path, output_dir="processed_data"):
    self.gdb_path = gdb_path
    self.output_dir = Path(output_dir)

# Usage:
processor = FireDataProcessor("../Forest_Fires/Bushfire_Boundaries_Historical_2024_V3.gdb")
```

**With:**
```python
# NEW
def __init__(self, gdb_path=None, output_dir=None):
    config = FirePrintConfig()
    self.gdb_path = gdb_path or str(config.get_path('source_data.bushfire_gdb'))
    self.output_dir = Path(output_dir) if output_dir else config.get_path('processed_data.demo')

# Usage:
processor = FireDataProcessor()  # Uses config automatically
```

**Replace save/load calls:**
```python
# OLD
save_processed_data(fingerprints, labels, metadata, encoders, "demo_processed_data")
loaded_data = load_processed_data("demo_processed_data")
```

**With:**
```python
# NEW
output_dir = str(config.get_path('processed_data.demo'))
save_processed_data(fingerprints, labels, metadata, encoders, output_dir)
loaded_data = load_processed_data(output_dir)
```

---

### Notebook 03: CNN Architecture and Training

**Add at the top:**
```python
from src.config_loader import FirePrintConfig
config = FirePrintConfig()
```

**Replace:**
```python
# OLD
def load_processed_data(data_dir="demo_processed_data"):
    ...

fingerprints, labels, metadata, encoders = load_processed_data("demo_processed_data")
trainer = FireCNNTrainer(model, task_names, model_save_path="demo_training_models")
```

**With:**
```python
# NEW
def load_processed_data(data_dir=None):
    if data_dir is None:
        config = FirePrintConfig()
        data_dir = str(config.get_path('processed_data.demo'))
    ...

fingerprints, labels, metadata, encoders = load_processed_data()
model_save_path = str(config.get_path('models.demo_training'))
trainer = FireCNNTrainer(model, task_names, model_save_path=model_save_path)
```

---

### Notebook 04: Pattern Analysis and Features

**Add at the top:**
```python
from src.config_loader import FirePrintConfig
config = FirePrintConfig()
```

**Replace:**
```python
# OLD
fingerprints, labels, metadata, encoders = load_processed_data("demo_processed_data")
features_df.to_csv('demo_fire_features.csv', index=False)
```

**With:**
```python
# NEW
data_dir = str(config.get_path('processed_data.demo'))
fingerprints, labels, metadata, encoders = load_processed_data(data_dir)

output_path = config.get_path('outputs.demo_features_csv', create=True)
features_df.to_csv(output_path, index=False)
```

---

### Notebook 05: Similarity Search and Clustering

**Add at the top:**
```python
from src.config_loader import FirePrintConfig
config = FirePrintConfig()
```

**Replace:**
```python
# OLD
fingerprints, labels, metadata, encoders = load_processed_data("demo_processed_data")
features_df = pd.read_csv('demo_fire_features.csv')
cnn_features = np.load('demo_cnn_features.npy')
save_similarity_search_engine(search_engine, "demo_similarity_search")
```

**With:**
```python
# NEW
data_dir = str(config.get_path('processed_data.demo'))
fingerprints, labels, metadata, encoders = load_processed_data(data_dir)

features_path = config.get_path('outputs.demo_features_csv')
features_df = pd.read_csv(features_path)

cnn_path = config.get_path('outputs.demo_cnn_features')
cnn_features = np.load(cnn_path)

search_dir = str(config.get_path('search.demo'))
save_similarity_search_engine(search_engine, search_dir)
```

---

### Notebook 06: Complete System Demo

**Add at the top:**
```python
from src.config_loader import FirePrintConfig
config = FirePrintConfig()
```

**Replace all hardcoded paths with config lookups** (same pattern as above)

---

## 🔧 Configuration Updates

### To Update the GDB Path

Edit `FirePrint-v1.0/config.yaml`:
```yaml
paths:
  source_data:
    bushfire_gdb: "../data/Bushfire_Boundaries_Historical_2024_V3.gdb"
    # Change to your path ↑
```

All notebooks will automatically use the new path!

### To Switch Between Demo and Production

In `config.yaml`:
```yaml
# For demo
processed_data:
  demo: "../data/demo_processed_data"
  
# For production
processed_data:
  main: "../data/processed_data"
```

In notebooks:
```python
# Use demo
data_dir = config.get_path('processed_data.demo')

# Use production
data_dir = config.get_path('processed_data.main')
```

---

## ✅ Benefits

1. **Update Once**: Change path in config.yaml, all notebooks updated
2. **No Hardcoding**: All paths centrally managed
3. **Portable**: Easy to move project to new location
4. **Documented**: All paths clearly named and explained
5. **Type Safe**: Path validation and resolution automatic

---

## 🚀 Testing the Migration

After updating a notebook:

1. **Restart kernel** to ensure clean state
2. **Run cells** that load the config
3. **Verify paths** are correct:
   ```python
   print(f"GDB: {config.get_path('source_data.bushfire_gdb')}")
   print(f"Data: {config.get_path('processed_data.demo')}")
   ```
4. **Test data loading** to ensure paths work
5. **Save a file** to test output paths

---

## 💡 Pro Tips

1. Keep one cell at the top that loads config and prints key paths
2. Use `create=True` when getting output directories
3. Convert Path objects to strings for compatibility: `str(config.get_path(...))`
4. Use the config for parameters too: `config.get('processing.image_size')`

---

## 📚 Resources

- **config.yaml**: Main configuration file
- **CONFIG_USAGE.md**: Detailed usage guide
- **examples/using_config.py**: Complete examples
- **src/config_loader.py**: Implementation

