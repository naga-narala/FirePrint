"""
Example: Using FirePrint Configuration in Notebooks
====================================================

This example demonstrates how to use the config.yaml file in your notebooks
to manage all paths and parameters centrally.

Benefits:
- Update paths in ONE place (config.yaml)
- All notebooks automatically use updated paths
- Easy to switch between demo/production environments
- Portable across different systems
"""

# ============================================================
# Example 1: Basic Configuration Loading
# ============================================================

from pathlib import Path
import sys

# Add src to path if running as script
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config_loader import FirePrintConfig

# Load configuration
config = FirePrintConfig()

print("=" * 60)
print("Example 1: Loading Configuration")
print("=" * 60)

# Access project info
print(f"Project: {config.get('project.name')}")
print(f"Version: {config.get('project.version')}")
print()

# ============================================================
# Example 2: Working with Paths
# ============================================================

print("=" * 60)
print("Example 2: Path Management")
print("=" * 60)

# Get source data path (GDB file)
gdb_path = config.get_path('source_data.bushfire_gdb')
print(f"GDB Path: {gdb_path}")
print(f"Exists: {gdb_path.exists()}")
print()

# Get processed data directory
demo_data_dir = config.get_path('processed_data.demo')
print(f"Demo Data Dir: {demo_data_dir}")
print(f"Exists: {demo_data_dir.exists()}")
print()

# Get model directory (and create if needed)
model_dir = config.get_path('models.demo_training', create=True)
print(f"Model Dir: {model_dir}")
print(f"Created: {model_dir.exists()}")
print()

# ============================================================
# Example 3: Complete File Paths
# ============================================================

print("=" * 60)
print("Example 3: Complete File Paths")
print("=" * 60)

# Get complete path for fingerprints file
fingerprints_path = config.get_file_path('processed_data.demo', 'fingerprints')
print(f"Fingerprints: {fingerprints_path}")

# Get complete path for model file
model_path = config.get_file_path('models.demo_training', 'best_model')
print(f"Best Model: {model_path}")

# Get complete path for features CSV
features_path = config.get_path('outputs.demo_features_csv')
print(f"Features CSV: {features_path}")
print()

# ============================================================
# Example 4: Processing Parameters
# ============================================================

print("=" * 60)
print("Example 4: Processing Parameters")
print("=" * 60)

# Get processing parameters
image_size = config.get('processing.image_size')
batch_size = config.get('processing.batch_size')
chunk_size = config.get('processing.chunk_size')

print(f"Image Size: {image_size}")
print(f"Batch Size: {batch_size}")
print(f"Chunk Size: {chunk_size}")
print()

# ============================================================
# Example 5: Model Configuration
# ============================================================

print("=" * 60)
print("Example 5: Model Configuration")
print("=" * 60)

# Get model parameters
architecture = config.get('model.architecture')
learning_rate = config.get('model.initial_learning_rate')
epochs = config.get('model.epochs')

print(f"Architecture: {architecture}")
print(f"Learning Rate: {learning_rate}")
print(f"Epochs: {epochs}")
print()

# Get task configuration
tasks = config.get('model.tasks')
print(f"Number of Tasks: {len(tasks)}")
for task in tasks:
    print(f"  - {task['name']} ({task['type']})")
print()

# ============================================================
# Example 6: Feature Configuration
# ============================================================

print("=" * 60)
print("Example 6: Feature Lists")
print("=" * 60)

# Get feature lists
geometric_features = config.get('features.geometric')
texture_features = config.get('features.texture')

print(f"Geometric Features ({len(geometric_features)}):")
print(f"  {', '.join(geometric_features[:5])}...")
print()
print(f"Texture Features ({len(texture_features)}):")
print(f"  {', '.join(texture_features)}")
print()

# ============================================================
# Example 7: Using in Notebook Context
# ============================================================

print("=" * 60)
print("Example 7: Typical Notebook Usage Pattern")
print("=" * 60)

# This is how you'd typically use it in a notebook
print("""
# At the top of your notebook:

from src.config_loader import FirePrintConfig
config = FirePrintConfig()

# Then throughout your notebook:

# Load processed data
data_dir = config.get_path('processed_data.demo')
fingerprints = np.load(data_dir / config.get('files.fingerprints'))
labels = pickle.load(open(data_dir / config.get('files.labels'), 'rb'))

# Save model
model_dir = config.get_path('models.demo_training', create=True)
model.save(model_dir / config.get('files.best_model'))

# Process with correct parameters
image_size = config.get('processing.image_size')
fingerprint = create_fingerprint(polygon, image_size=image_size)
""")

# ============================================================
# Example 8: Updating Paths Programmatically
# ============================================================

print("=" * 60)
print("Example 8: Updating Configuration")
print("=" * 60)

# Update a path (this saves to config.yaml)
# Uncomment to actually update:
# config.update_path('source_data.bushfire_gdb', 
#                    '/new/path/to/Bushfire_Boundaries.gdb')

print("To update paths programmatically:")
print("  config.update_path('source_data.bushfire_gdb', 'new/path.gdb')")
print()
print("Or simply edit config.yaml directly!")
print()

# ============================================================
# Example 9: Environment-Specific Configuration
# ============================================================

print("=" * 60)
print("Example 9: Environment Information")
print("=" * 60)

conda_env = config.get('environment.conda_env')
python_version = config.get('environment.python_version')

print(f"Recommended Conda Environment: {conda_env}")
print(f"Python Version: {python_version}")
print()

# ============================================================
# Summary
# ============================================================

print("=" * 60)
print("Summary: Benefits of Using config.yaml")
print("=" * 60)
print("""
✅ Central Configuration: All paths in one place
✅ Easy Updates: Change once, applies everywhere
✅ Portable: Works across different systems
✅ Documented: Parameters are clearly named and explained
✅ Version Control: Track configuration changes in git
✅ Type Safety: Automatic path resolution and validation

To use in your notebooks:
1. Import: from src.config_loader import FirePrintConfig
2. Load: config = FirePrintConfig()
3. Use: path = config.get_path('processed_data.demo')

To update paths:
- Edit FirePrint-v1.0/config.yaml
- All notebooks automatically use new paths!
""")
print("=" * 60)

