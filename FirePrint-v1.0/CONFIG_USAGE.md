# FirePrint Configuration System

## 📋 Overview

The FirePrint configuration system centralizes all paths and parameters in a single `config.yaml` file. This makes it easy to update paths across all notebooks and ensures consistency throughout the project.

## 🚀 Quick Start

### In Notebooks

Add this to the top of your notebook:

```python
from src.config_loader import FirePrintConfig

# Load configuration
config = FirePrintConfig()

# Use paths
gdb_path = config.get_path('source_data.bushfire_gdb')
demo_data_dir = config.get_path('processed_data.demo')
model_dir = config.get_path('models.demo_training')

# Use parameters
image_size = config.get('processing.image_size')
batch_size = config.get('model.batch_size')
```

## 📁 Common Path Access Patterns

### Source Data
```python
# Get GDB path
gdb_path = config.get_path('source_data.bushfire_gdb')
layer_name = config.get('paths.source_data.bushfire_layer')
```

### Processed Data
```python
# Get directory
data_dir = config.get_path('processed_data.demo')

# Get complete file paths
fingerprints_path = config.get_file_path('processed_data.demo', 'fingerprints')
labels_path = config.get_file_path('processed_data.demo', 'labels')
metadata_path = config.get_file_path('processed_data.demo', 'metadata')
```

### Models
```python
# Get model directory (create if needed)
model_dir = config.get_path('models.demo_training', create=True)

# Get specific model files
best_model = config.get_file_path('models.demo_training', 'best_model')
trained_model = config.get_file_path('models.demo_training', 'trained_model')
```

### Outputs
```python
# Output paths
features_csv = config.get_path('outputs.demo_features_csv')
cnn_features = config.get_path('outputs.demo_cnn_features')
```

## ⚙️ Common Parameters

### Processing
```python
image_size = config.get('processing.image_size')  # 224
batch_size = config.get('processing.batch_size')  # 32
chunk_size = config.get('processing.chunk_size')  # 10000
```

### Model
```python
architecture = config.get('model.architecture')  # "EfficientNetB0"
learning_rate = config.get('model.initial_learning_rate')  # 0.001
epochs = config.get('model.epochs')  # 50
tasks = config.get('model.tasks')  # List of task configurations
```

### Features
```python
geometric_features = config.get('features.geometric')
texture_features = config.get('features.texture')
```

## 🔄 Updating Configuration

### Method 1: Edit config.yaml Directly (Recommended)

Simply open `FirePrint-v1.0/config.yaml` and edit the values:

```yaml
paths:
  source_data:
    bushfire_gdb: "../data/Bushfire_Boundaries_Historical_2024_V3.gdb"
    # Change to your path ↑
```

### Method 2: Programmatically

```python
config.update_path('source_data.bushfire_gdb', '/new/path/to/data.gdb')
```

## 📝 Notebook Migration Guide

### Before (Old Way)
```python
# Hardcoded paths
gdb_path = "../Forest_Fires/Bushfire_Boundaries_Historical_2024_V3.gdb"
data_dir = "demo_processed_data"
model_dir = "demo_training_models"
image_size = 224
```

### After (New Way)
```python
# Load from config
from src.config_loader import FirePrintConfig
config = FirePrintConfig()

gdb_path = config.get_path('source_data.bushfire_gdb')
data_dir = config.get_path('processed_data.demo')
model_dir = config.get_path('models.demo_training')
image_size = config.get('processing.image_size')
```

## 🎯 Benefits

### ✅ Central Management
- All paths in one file
- No more searching through notebooks

### ✅ Easy Updates
- Change path once
- All notebooks automatically updated

### ✅ Portability
- Move project to new location
- Update one path in config.yaml
- Everything works!

### ✅ Documentation
- All parameters clearly named
- Default values documented
- Easy to understand project structure

### ✅ Version Control
- Track configuration changes
- See what changed over time
- Easy to revert if needed

## 📖 Full Example

See `examples/using_config.py` for complete examples of all features.

## 🔧 Advanced Usage

### Get All Paths
```python
all_paths = config.get_all_paths()
for key, path in all_paths.items():
    print(f"{key}: {path}")
```

### Print Configuration
```python
# Print entire config
config.print_config()

# Print specific section
config.print_config('paths')
config.print_config('model')
```

### Default Values
```python
# Use default if key doesn't exist
value = config.get('some.key', default=42)
```

## 🗂️ Configuration Structure

```
config.yaml
├── project: Project metadata
├── paths: All directory and file paths
│   ├── source_data: Input data locations
│   ├── processed_data: Processed outputs
│   ├── models: Model directories
│   ├── search: Search engine directories
│   └── outputs: Output file paths
├── files: Standard filenames
├── processing: Processing parameters
├── model: Model architecture and training
├── features: Feature lists
├── similarity: Similarity search config
├── visualization: Plot settings
├── logging: Logging configuration
└── environment: Environment requirements
```

## 💡 Tips

1. **Always use `config.get_path()` for paths** - it handles resolution automatically
2. **Use `create=True` when saving files** - creates directories if needed
3. **Edit config.yaml for one-time changes** - simpler than programmatic updates
4. **Keep config.yaml in version control** - track your configuration changes
5. **Use `config.get()` with defaults** - prevents errors if key is missing

## 🆘 Troubleshooting

### Config file not found
```python
# Specify path explicitly
config = FirePrintConfig('path/to/config.yaml')
```

### Path doesn't exist
```python
# Create directory when getting path
path = config.get_path('models.demo_training', create=True)
```

### Import error
```python
# Make sure you're in the right directory or add to path
import sys
from pathlib import Path
sys.path.append(str(Path('FirePrint-v1.0/src')))
from config_loader import FirePrintConfig
```

## 📚 Reference

- **config.yaml**: Main configuration file
- **src/config_loader.py**: Configuration loader implementation
- **examples/using_config.py**: Complete usage examples
- **CONFIG_USAGE.md**: This file!

