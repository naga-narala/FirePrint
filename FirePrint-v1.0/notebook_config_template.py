"""
Notebook Configuration Template
================================
Copy this cell to the top of your notebooks to use the config system!
"""

# ==================================================
# CONFIGURATION SETUP - Add this to your notebooks
# ==================================================

from pathlib import Path
import sys

# Add src to path (if not already there)
notebook_dir = Path.cwd()
src_path = notebook_dir.parent / 'src' if 'notebooks' in str(notebook_dir) else notebook_dir / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import config loader
from config_loader import FirePrintConfig

# Load configuration
config = FirePrintConfig()

# Print configuration info
print("=" * 60)
print("🔥 FirePrint Configuration Loaded")
print("=" * 60)
print(f"Project: {config.get('project.name')} v{config.get('project.version')}")
print()
print("Key Paths:")
print(f"  GDB: {config.get_path('source_data.bushfire_gdb')}")
print(f"  Data: {config.get_path('processed_data.demo')}")
print(f"  Models: {config.get_path('models.demo_training')}")
print()
print("Parameters:")
print(f"  Image Size: {config.get('processing.image_size')}")
print(f"  Batch Size: {config.get('processing.batch_size')}")
print("=" * 60)

# ==================================================
# USAGE EXAMPLES
# ==================================================

# Get paths
gdb_path = str(config.get_path('source_data.bushfire_gdb'))
data_dir = str(config.get_path('processed_data.demo'))
model_dir = str(config.get_path('models.demo_training', create=True))

# Get file paths
fingerprints_path = config.get_file_path('processed_data.demo', 'fingerprints')
labels_path = config.get_file_path('processed_data.demo', 'labels')
best_model_path = config.get_file_path('models.demo_training', 'best_model')

# Get parameters
image_size = config.get('processing.image_size')
batch_size = config.get('processing.batch_size')
learning_rate = config.get('model.initial_learning_rate')

# Get feature lists
geometric_features = config.get('features.geometric')
texture_features = config.get('features.texture')

# ==================================================
# COMPATIBILITY FUNCTIONS (for existing code)
# ==================================================

def load_processed_data(data_dir=None):
    """Load processed data with automatic path resolution"""
    import pickle
    import numpy as np
    import json
    
    if data_dir is None:
        data_dir = config.get_path('processed_data.demo')
    else:
        data_dir = Path(data_dir)
    
    # Load fingerprints
    fingerprints = np.load(data_dir / config.get('files.fingerprints'))
    
    # Load labels
    with open(data_dir / config.get('files.labels'), 'rb') as f:
        labels = pickle.load(f)
    
    # Load metadata
    with open(data_dir / config.get('files.metadata'), 'rb') as f:
        metadata = pickle.load(f)
    
    # Load encoders
    with open(data_dir / config.get('files.encoders'), 'r') as f:
        encoders = json.load(f)
    
    return fingerprints, labels, metadata, encoders


def save_processed_data(fingerprints, labels, metadata, encoders, output_dir=None):
    """Save processed data with automatic path resolution"""
    import pickle
    import numpy as np
    import json
    
    if output_dir is None:
        output_dir = config.get_path('processed_data.demo', create=True)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save fingerprints
    np.save(output_dir / config.get('files.fingerprints'), fingerprints)
    
    # Save labels
    with open(output_dir / config.get('files.labels'), 'wb') as f:
        pickle.dump(labels, f)
    
    # Save metadata
    with open(output_dir / config.get('files.metadata'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save encoders
    with open(output_dir / config.get('files.encoders'), 'w') as f:
        json.dump(encoders, f, indent=2)
    
    print(f"✓ Processed data saved to {output_dir}")


# ==================================================
# Ready to use! Example:
# ==================================================
# fingerprints, labels, metadata, encoders = load_processed_data()
# print(f"Loaded {len(fingerprints)} fingerprints")
# ==================================================

