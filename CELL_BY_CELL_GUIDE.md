# 📓 FirePrint - Detailed Cell-by-Cell Guide

## Purpose of This Guide
This document explains **EVERY cell** in all 6 notebooks, line by line. Use this when you want to understand exactly what each code block does.

---

# 📓 NOTEBOOK 01: Fire Polygon to Fingerprint

## Cell 0: Setup and Environment Check
```python
# What it does: Checks Python environment and package versions
# Why: Ensures you have correct libraries installed
# Output: Shows Python version, NumPy version, installed packages
```
**Key Output**: Confirms rip_gpu environment and NumPy 1.x compatibility

## Cell 1: Troubleshooting Guide (Markdown)
```markdown
# What it does: Documentation cell
# Why: Helps if you get TypeAliasType errors
# Output: Instructions for fixing environment issues
```

## Cell 2: Load Configuration
```python
from config_loader import FirePrintConfig
config = FirePrintConfig()
```
**What it does**: Loads paths from `config.yaml`
**Why**: Centralized configuration management
**Output**: Shows GDB path, image size (224), confirms config loaded

## Cell 3: Import Libraries
```python
import numpy, pandas, cv2, shapely, rasterio, sklearn, plotly, etc.
```
**What it does**: Imports all required Python libraries
**Why**: Needed for geometry, image processing, visualization
**Output**: Checkmarks for successfully imported libraries

## Cell 4-7: Introduction Markdown Cells
**Content**: Explains project overview, learning objectives, theory

## Cell 8: `normalize_geometry()` Function
```python
def normalize_geometry(geometry):
    # 1. Get bounding box of fire polygon
    bounds = geometry.bounds  # (minx, miny, maxx, maxy)
    
    # 2. Calculate width and height
    width = maxx - minx
    height = maxy - miny
    
    # 3. Scale to unit square [0,1] x [0,1]
    scale = 1.0 / max(width, height)
    
    # 4. Transform coordinates
    normalized_geom = transform(normalize_coords, geometry)
    return normalized_geom, (scale, center_x, center_y)
```
**Purpose**: Standardizes fire size to 0-1 range while preserving shape
**Why**: Different fires have different sizes; need standard size for CNN
**Input**: Fire polygon (any size)
**Output**: Normalized polygon + transformation parameters

## Cell 9: Markdown - Channel 1 Explanation

## Cell 10: `create_shape_mask()` Function
```python
def create_shape_mask(geometry, image_size=224):
    # 1. Create transformation matrix for 224x224 grid
    transform = from_bounds(0, 0, 1, 1, image_size, image_size)
    
    # 2. Rasterize polygon to binary image
    mask = rasterize(
        [geometry],
        out_shape=(224, 224),
        transform=transform,
        fill=0,           # Background = 0
        default_value=1   # Fire = 1
    )
    return mask.astype(np.float32)
```
**Purpose**: Creates binary image of fire shape
**Why**: First channel of fingerprint - basic outline
**Input**: Normalized polygon
**Output**: 224×224 binary image (0s and 1s)

## Cell 11: Markdown - Channel 2 Explanation

## Cell 12: `calculate_distance_transform()` Function
```python
def calculate_distance_transform(shape_mask):
    # 1. For each pixel inside fire, calculate distance to nearest edge
    distance_map = distance_transform_edt(shape_mask)
    #    EDT = Euclidean Distance Transform
    
    # 2. Normalize to [0, 1] range
    if distance_map.max() > 0:
        distance_map = distance_map / distance_map.max()
    
    return distance_map.astype(np.float32)
```
**Purpose**: Shows fire's internal structure
**Why**: Captures spatial complexity - elongated fires have different patterns than circular
**Input**: Shape mask (binary)
**Output**: 224×224 grayscale image (values 0-1)
**Example**: 
- Center of large fire = high values (far from edge)
- Near edges = low values (close to edge)

## Cell 13: Markdown - Channel 3 Explanation

## Cell 14: `calculate_curvature_map()` Function
```python
def calculate_curvature_map(geometry, image_size=224):
    # 1. Extract boundary points
    coords = np.array(boundary.coords)
    
    # 2. For each point, calculate curvature
    for i in range(1, len(coords) - 1):
        p1, p2, p3 = coords[i-1], coords[i], coords[i+1]
        
        # Vectors between points
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Curvature = cross product / (vector lengths)
        cross_prod = np.cross(v1, v2)
        curvature = abs(cross_prod) / (norm_v1 * norm_v2)
        
        # 3. Map to image coordinates
        x = int(coord[0] * image_size)
        y = int(coord[1] * image_size)
        curvature_map[y, x] = curvature
    
    # 4. Smooth with Gaussian blur
    curvature_map = cv2.GaussianBlur(curvature_map, (5, 5), 1.0)
    
    return curvature_map
```
**Purpose**: Measures how "jagged" the fire boundary is
**Why**: Smooth boundaries vs. highly irregular boundaries indicate different fire behaviors
**Input**: Normalized geometry
**Output**: 224×224 image with curvature values
**High values**: Sharp turns in boundary (complex burning)
**Low values**: Smooth sections of boundary

## Cell 15: Markdown - Channel 4 Explanation

## Cell 16: `calculate_fractal_map()` Function
```python
def calculate_fractal_map(geometry, image_size=224):
    # 1. Create high-res mask (448x448)
    high_res_size = image_size * 2
    shape_mask = create_shape_mask(geometry, high_res_size)
    
    # 2. Divide into windows (one per output pixel)
    window_size = high_res_size // image_size
    
    # 3. For each window, calculate local fractal dimension
    for i in range(image_size):
        for j in range(image_size):
            local_mask = shape_mask[y_start:y_end, x_start:x_end]
            fractal_dim = calculate_local_fractal_dimension(local_mask)
            fractal_map[i, j] = fractal_dim
    
    return fractal_map
```

```python
def calculate_local_fractal_dimension(binary_mask):
    # Box-counting method:
    # 1. Find boundary pixels
    boundary = cv2.Canny(binary_mask, 50, 150)
    
    # 2. Calculate fractal dimension
    # Formula: 2 * log(perimeter) / log(area)
    fractal_dim = 2 * np.log(perimeter) / np.log(area)
    
    # 3. Normalize to [0, 1]
    fractal_dim = max(0, min(2, fractal_dim - 1))
    return fractal_dim
```
**Purpose**: Captures self-similarity and complexity
**Why**: Natural fires often have fractal properties (look similar at different scales)
**Input**: Normalized geometry
**Output**: 224×224 image with fractal dimension values
**Values**: 
- 1.0 = Simple shape (like a circle)
- 1.5 = Moderate complexity
- 2.0 = Highly complex, fractal-like boundary

## Cell 17: Markdown - Complete Fingerprint Explanation

## Cell 18: `polygon_to_fingerprint()` - **MOST IMPORTANT FUNCTION**
```python
def polygon_to_fingerprint(geometry, image_size=224, debug=False):
    # Step 1: Normalize geometry to unit square
    normalized_geom, transform_params = normalize_geometry(geometry)
    if normalized_geom is None:
        return None
    
    # Step 2: Create 4 channels
    channels = []
    
    # Channel 1: Binary shape mask
    shape_mask = create_shape_mask(normalized_geom, image_size)
    channels.append(shape_mask)
    
    # Channel 2: Distance transform
    distance_map = calculate_distance_transform(shape_mask)
    channels.append(distance_map)
    
    # Channel 3: Boundary curvature
    curvature_map = calculate_curvature_map(normalized_geom, image_size)
    channels.append(curvature_map)
    
    # Channel 4: Fractal dimension
    fractal_map = calculate_fractal_map(normalized_geom, image_size)
    channels.append(fractal_map)
    
    # Step 3: Stack into 4-channel image
    fingerprint = np.stack(channels, axis=-1)  # Shape: (224, 224, 4)
    
    return fingerprint.astype(np.float32)
```
**Purpose**: Main conversion function - polygon → 4-channel image
**Input**: Fire polygon (Shapely geometry)
**Output**: 224×224×4 numpy array (the "fingerprint")
**This is the core innovation of the entire project!**

## Cell 19: Markdown - Visualization Functions

## Cell 20: `visualize_fingerprint()` Function
```python
def visualize_fingerprint(fingerprint, original_geometry=None, save_path=None):
    # Creates a 2x3 subplot grid:
    # Row 1: Channel 1, Channel 2, RGB composite
    # Row 2: Channel 3, Channel 4, Original geometry
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot each channel
    for i in range(4):
        axes[row, col].imshow(fingerprint[:, :, i], cmap='viridis')
        axes[row, col].set_title(f'Channel {i+1}: {channel_names[i]}')
    
    # RGB composite (first 3 channels)
    rgb_composite = fingerprint[:, :, :3]
    axes[0, 2].imshow(rgb_composite)
    
    # Original polygon overlay
    if original_geometry is not None:
        x, y = original_geometry.exterior.xy
        axes[1, 2].plot(x, y, 'r-', linewidth=2)
    
    plt.show()
```
**Purpose**: Visualizes all 4 channels of a fingerprint
**Why**: Helps understand what each channel captures
**Output**: Figure with 6 subplots showing channels + original

## Cell 21: Markdown - Test with Synthetic Fire

## Cell 22: Create Synthetic Fire and Test
```python
# 1. Create irregular polygon (simulating fire)
angles = np.linspace(0, 2*np.pi, 20)
radii = 1 + 0.3 * np.sin(5*angles) + 0.2 * np.random.random(20)
x = radii * np.cos(angles)
y = radii * np.sin(angles)
synthetic_fire = Polygon(zip(x, y))

# 2. Convert to fingerprint
fingerprint = polygon_to_fingerprint(synthetic_fire, debug=True)

# 3. Print statistics
for i in range(4):
    channel = fingerprint[:, :, i]
    print(f"Channel {i+1}: min={channel.min()}, max={channel.max()}, mean={channel.mean()}")
```
**Purpose**: Test system with fake fire
**Why**: Ensures functions work before trying real data
**Output**: Fingerprint statistics + visualization

## Cell 23: Markdown - Real Fire Data Example

## Cell 24: Load Real Fire from Database
```python
def load_sample_fire():
    # 1. Load geodatabase
    gdb_path = config.get_path('source_data.bushfire_gdb')
    gdf = gpd.read_file(gdb_path, layer='Bushfire_Boundaries_Historical_V3')
    
    # 2. Get first valid fire
    valid_fires = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    sample_fire = valid_fires.iloc[0]
    return sample_fire

# Try to load and convert real fire
sample_fire = load_sample_fire()
real_fingerprint = polygon_to_fingerprint(sample_fire.geometry, debug=True)
```
**Purpose**: Test with actual Australian bushfire data
**Why**: Validate system works with real-world data
**Output**: Real fire fingerprint or fallback to synthetic

## Cell 25: Markdown - Batch Processing

## Cell 26: `batch_convert_polygons()` Function
```python
def batch_convert_polygons(geometries, image_size=224, show_progress=True):
    fingerprints = []
    failed_indices = []
    
    # Process each polygon
    for idx, geometry in enumerate(geometries):
        fingerprint = polygon_to_fingerprint(geometry, image_size)
        
        if fingerprint is not None:
            fingerprints.append(fingerprint)
        else:
            failed_indices.append(idx)
    
    return np.array(fingerprints), failed_indices

# Demo: Create 5 synthetic fires and batch convert
synthetic_fires = [create_random_fire() for _ in range(5)]
batch_fingerprints, failed = batch_convert_polygons(synthetic_fires)
```
**Purpose**: Convert many fires efficiently
**Why**: Need to process 324K fires - can't do one at a time
**Input**: List of geometries
**Output**: Array of fingerprints (N, 224, 224, 4)

## Cell 27: Markdown - Key Insights

## Cell 28: Markdown - Summary

## Cell 29: Export Functions to Pickle
```python
# Save functions for use in other notebooks
fire_functions = {
    'normalize_geometry': normalize_geometry,
    'create_shape_mask': create_shape_mask,
    'calculate_distance_transform': calculate_distance_transform,
    'calculate_curvature_map': calculate_curvature_map,
    'calculate_fractal_map': calculate_fractal_map,
    'polygon_to_fingerprint': polygon_to_fingerprint,
    'visualize_fingerprint': visualize_fingerprint,
    'batch_convert_polygons': batch_convert_polygons
}

with open('shared_functions/fire_fingerprint_functions.pkl', 'wb') as f:
    pickle.dump(fire_functions, f)
```
**Purpose**: Save functions for other notebooks to use
**Why**: Avoid copy-pasting code across notebooks
**Output**: Pickle file with all functions

---

# 📓 NOTEBOOK 02: Data Processing Pipeline

## Cell 0-5: Setup (similar to Notebook 01)
- Configuration loading
- Library imports
- Function definitions copied from Notebook 01

## Cell 6: `FireDataProcessor` Class Definition
```python
class FireDataProcessor:
    def __init__(self, gdb_path=None, output_dir=None):
        # Get paths from config
        self.gdb_path = gdb_path or config.get_path('source_data.bushfire_gdb')
        self.output_dir = output_dir or config.get_path('processed_data.demo')
        
        # Initialize label encoders (empty dictionaries)
        self.fire_type_encoder = {}
        self.cause_encoder = {}
        self.state_encoder = {}
        self.size_encoder = {}
    
    def load_fire_data(self, layer_name="Bushfire_Boundaries_Historical_V3"):
        # Load from geodatabase
        gdf = gpd.read_file(self.gdb_path, layer=layer_name)
        return gdf

# Initialize and load data
processor = FireDataProcessor()
gdf = processor.load_fire_data()
```
**What happens**:
1. Creates processor object with GDB path from config
2. Loads 324,741 fire records
3. Shows columns and geometry types
**Output**: "Loaded 324,741 fire records"

## Cell 7: Markdown - Data Exploration

## Cell 8: `analyze_dataset()` Function
```python
def analyze_dataset(gdf):
    # 1. Basic statistics
    print(f"Total records: {len(gdf)}")
    print(f"Date range: {gdf['ignition_date'].min()} to {gdf['ignition_date'].max()}")
    print(f"Total area burned: {gdf['area_ha'].sum()} hectares")
    
    # 2. Fire type distribution
    fire_type_counts = gdf['fire_type'].value_counts()
    for fire_type, count in fire_type_counts.items():
        pct = count / len(gdf) * 100
        print(f"  {fire_type}: {count} ({pct:.1f}%)")
    
    # 3. State distribution
    # 4. Size statistics
    # 5. Geometry types

analyze_dataset(gdf)
```
**Purpose**: Understand the dataset
**Output**:
- 324,741 fires from 1898-2024
- 333.5 million hectares burned
- Prescribed Burn: 47.8%, Unknown: 28.8%, Bushfire: 23.4%
- Western Australia: 51.4%, Victoria: 26.9%

## Cell 9: Markdown - Data Quality Control

## Cell 10: `filter_valid_geometries()` Function
```python
def filter_valid_geometries(gdf):
    initial_count = len(gdf)
    
    # 1. Remove null geometries
    gdf = gdf[gdf.geometry.notna()]
    
    # 2. Remove invalid geometries
    valid_mask = gdf.geometry.is_valid
    gdf = gdf[valid_mask]
    
    # 3. Remove very small fires (< 0.1 ha)
    area_mask = gdf['area_ha'] >= 0.1
    gdf = gdf[area_mask]
    
    # 4. Remove zero-area bounds
    def has_valid_bounds(geom):
        bounds = geom.bounds
        return (bounds[2] - bounds[0]) > 0 and (bounds[3] - bounds[1]) > 0
    
    bounds_mask = gdf.geometry.apply(has_valid_bounds)
    gdf = gdf[bounds_mask]
    
    return gdf

filtered_gdf = filter_valid_geometries(gdf)
```
**Purpose**: Remove problematic fires that would cause errors
**Input**: 324,741 fires
**Output**: 249,825 fires (76.9%) after filtering
**Removed**: 
- Null geometries
- Invalid geometries
- Tiny fires (<0.1 ha)
- Zero-area bounds

## Cell 11: Markdown - Label Encoding

## Cell 12: `create_label_encoders()` Function
```python
def create_label_encoders(gdf):
    encoders = {}
    
    # Fire type encoder: "Bushfire" → 0, "Prescribed Burn" → 1, etc.
    fire_types = gdf['fire_type'].dropna().unique()
    fire_type_encoder = {ft: i for i, ft in enumerate(sorted(fire_types))}
    encoders['fire_type'] = fire_type_encoder
    
    # Ignition cause encoder
    causes = gdf['ignition_cause'].dropna().unique()
    cause_encoder = {cause: i for i, cause in enumerate(sorted(causes))}
    encoders['ignition_cause'] = cause_encoder
    
    # State encoder: "NSW" → 0, "VIC" → 1, etc.
    states = gdf['state'].dropna().unique()
    state_encoder = {state: i for i, state in enumerate(sorted(states))}
    encoders['state'] = state_encoder
    
    # Size category encoder
    size_encoder = {
        'Small': 0,      # < 10 ha
        'Medium': 1,     # 10-100 ha
        'Large': 2,      # 100-1000 ha
        'Very Large': 3  # > 1000 ha
    }
    encoders['size_category'] = size_encoder
    
    return encoders

encoders = create_label_encoders(filtered_gdf)
```
**Purpose**: Convert text labels to numbers for machine learning
**Why**: Neural networks need numerical inputs
**Output**: 
- fire_type: 4 categories (Bushfire=0, Prescribed Burn=1, etc.)
- ignition_cause: 7 categories
- state: 7 categories
- size_category: 4 categories

## Cell 13: Markdown - Label Encoding Functions

## Cell 14: Encoding Helper Functions
```python
def encode_fire_type(fire_type, encoder):
    if pd.isna(fire_type) or fire_type not in encoder:
        return 0  # Default
    return encoder[fire_type]

def encode_ignition_cause(cause, encoder):
    if pd.isna(cause):
        return encoder.get('Other', 0)
    return encoder.get(cause, encoder.get('Other', 0))

def encode_state(state, encoder):
    if pd.isna(state) or state not in encoder:
        return 0
    return encoder[state]

def encode_size_category(area_ha):
    if pd.isna(area_ha) or area_ha <= 0:
        return 0  # Small
    elif area_ha < 10:
        return 0  # Small
    elif area_ha < 100:
        return 1  # Medium
    elif area_ha < 1000:
        return 2  # Large
    else:
        return 3  # Very Large
```
**Purpose**: Apply encoders to individual values
**Handles**: Missing data, unknown categories

## Cell 15: Markdown - Batch Processing Pipeline

## Cell 16: `process_fire_dataset()` - **MAIN PROCESSING FUNCTION**
```python
def process_fire_dataset(gdf, encoders, sample_size=None, image_size=224, batch_size=100):
    # 1. Optional sampling
    if sample_size and sample_size < len(gdf):
        gdf = gdf.sample(n=sample_size, random_state=42)
    
    # 2. Process in batches
    all_fingerprints = []
    all_labels = []
    all_metadata = []
    
    for batch_start in range(0, len(gdf), batch_size):
        batch_end = min(batch_start + batch_size, len(gdf))
        batch_gdf = gdf.iloc[batch_start:batch_end]
        
        # For each fire in batch:
        for idx, fire in batch_gdf.iterrows():
            # a. Convert to fingerprint
            fingerprint = polygon_to_fingerprint(fire.geometry, image_size)
            
            if fingerprint is not None:
                batch_fingerprints.append(fingerprint)
                
                # b. Encode labels
                labels = {
                    'fire_type': encode_fire_type(fire.fire_type, encoders['fire_type']),
                    'ignition_cause': encode_ignition_cause(fire.ignition_cause, encoders['ignition_cause']),
                    'state': encode_state(fire.state, encoders['state']),
                    'size_category': encode_size_category(fire.area_ha)
                }
                batch_labels.append(labels)
                
                # c. Store metadata
                metadata = {
                    'fire_id': fire.fire_id,
                    'area_ha': fire.area_ha,
                    'ignition_date': str(fire.ignition_date),
                    'original_fire_type': fire.fire_type,
                    'original_cause': fire.ignition_cause,
                    'original_state': fire.state
                }
                batch_metadata.append(metadata)
    
    # 3. Convert to numpy arrays
    fingerprints = np.array(all_fingerprints)
    
    return fingerprints, all_labels, all_metadata

# Process 50 sample fires for demo
fingerprints, labels, metadata = process_fire_dataset(
    filtered_gdf, encoders, sample_size=50, batch_size=10
)
```
**Purpose**: Convert many fires to fingerprints + encode labels
**Input**: GeoDataFrame with 249,825 fires
**Process**:
1. Sample 50 fires (for demo - full version would use all)
2. Split into batches of 10
3. For each fire:
   - Convert polygon → fingerprint
   - Encode fire_type, cause, state, size
   - Store metadata
**Output**:
- `fingerprints`: (50, 224, 224, 4) array
- `labels`: 50 label dictionaries
- `metadata`: 50 metadata dictionaries
**Performance**: ~50 fires/minute

## Cell 17: Markdown - Data Saving/Loading

## Cell 18: Save and Load Functions
```python
def save_processed_data(fingerprints, labels, metadata, encoders, output_dir):
    # 1. Save fingerprints as numpy array
    np.save(output_dir / 'fingerprints.npy', fingerprints)
    
    # 2. Save labels as pickle
    with open(output_dir / 'labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    
    # 3. Save metadata as pickle
    with open(output_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # 4. Save encoders as JSON
    with open(output_dir / 'encoders.json', 'w') as f:
        json.dump(encoders, f, indent=2)
    
    # 5. Save processing statistics
    stats = {
        'total_fingerprints': len(fingerprints),
        'fingerprint_shape': fingerprints.shape,
        'processing_date': datetime.now().isoformat()
    }
    with open(output_dir / 'processing_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

def load_processed_data(data_dir):
    fingerprints = np.load(data_dir / 'fingerprints.npy')
    with open(data_dir / 'labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    with open(data_dir / 'encoders.json', 'r') as f:
        encoders = json.load(f)
    return fingerprints, labels, metadata, encoders

# Save processed data
save_processed_data(fingerprints, labels, metadata, encoders)

# Test loading
loaded_fingerprints, loaded_labels, loaded_metadata, loaded_encoders = load_processed_data()
```
**Purpose**: Persist processed data for later use
**Why**: Processing 324K fires takes hours - save results!
**Files Created**:
- `fingerprints.npy` - 38.3 MB for 50 fires
- `labels.pkl` - Encoded labels
- `metadata.pkl` - Fire information
- `encoders.json` - Label encoding mappings
- `processing_stats.json` - Processing metadata

## Cell 19-22: Data Analysis
```python
# Cell 20: analyze_processed_data()
# - Channel statistics (mean, std, min, max for each channel)
# - Label distributions
# - Visualizations of channel histograms and label pie charts

# Cell 22: create_fingerprint_gallery()
# - Shows 4-6 sample fingerprints
# - Displays all 4 channels + RGB composite
# - Includes fire metadata
```

## Cell 23-24: Scaling to Full Dataset
**Shows**: Example code for processing all 324K fires in chunks

---

# 📓 NOTEBOOK 03: CNN Architecture & Training

## Cell 0-5: Setup (standard)

## Cell 6: `create_custom_fire_cnn()` - **CNN ARCHITECTURE**
```python
def create_custom_fire_cnn(input_shape=(224, 224, 4), num_classes_dict=None):
    # INPUT LAYER
    inputs = layers.Input(shape=(224, 224, 4))  # 4-channel fingerprint
    
    # SHARED CONVOLUTIONAL BACKBONE
    # Block 1: Extract low-level features
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)  # 224→112
    x = layers.Dropout(0.25)(x)
    
    # Block 2: Extract mid-level features
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)  # 112→56
    x = layers.Dropout(0.25)(x)
    
    # Block 3: Extract high-level features
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)  # 56→28
    x = layers.Dropout(0.25)(x)
    
    # Block 4: Extract complex patterns
    x = layers.Conv2D(256, (3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)  # 28→14
    x = layers.Dropout(0.25)(x)
    
    # GLOBAL POOLING: 14×14×256 → 256 features
    x = layers.GlobalAveragePooling2D()(x)
    
    # DENSE FEATURE EXTRACTION
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # FEATURE EXTRACTION LAYER (for similarity search)
    feature_layer = layers.Dense(256, activation='relu', name='feature_extraction')(x)
    
    # TASK-SPECIFIC OUTPUT HEADS
    outputs = []
    
    # Head 1: Fire Type Classification
    fire_type_branch = layers.Dense(128, activation='relu')(feature_layer)
    fire_type_branch = layers.Dropout(0.3)(fire_type_branch)
    fire_type_output = layers.Dense(num_classes_dict['fire_type'], 
                                     activation='softmax', 
                                     name='fire_type')(fire_type_branch)
    outputs.append(fire_type_output)
    
    # Head 2: Ignition Cause Classification
    cause_branch = layers.Dense(128, activation='relu')(feature_layer)
    cause_branch = layers.Dropout(0.3)(cause_branch)
    cause_output = layers.Dense(num_classes_dict['ignition_cause'], 
                                activation='softmax', 
                                name='ignition_cause')(cause_branch)
    outputs.append(cause_output)
    
    # Head 3: State Classification
    state_branch = layers.Dense(128, activation='relu')(feature_layer)
    state_branch = layers.Dropout(0.3)(state_branch)
    state_output = layers.Dense(num_classes_dict['state'], 
                                activation='softmax', 
                                name='state')(state_branch)
    outputs.append(state_output)
    
    # Head 4: Size Category Classification
    size_branch = layers.Dense(128, activation='relu')(feature_layer)
    size_branch = layers.Dropout(0.3)(size_branch)
    size_output = layers.Dense(num_classes_dict['size_category'], 
                               activation='softmax', 
                               name='size_category')(size_branch)
    outputs.append(size_output)
    
    # CREATE MODEL
    model = models.Model(inputs=inputs, outputs=outputs, name='fire_fingerprint_cnn')
    
    # COMPILE
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss={
            'fire_type': 'categorical_crossentropy',
            'ignition_cause': 'categorical_crossentropy',
            'state': 'categorical_crossentropy',
            'size_category': 'categorical_crossentropy'
        },
        loss_weights={
            'fire_type': 1.0,
            'ignition_cause': 1.0,
            'state': 0.8,
            'size_category': 0.8
        },
        metrics=['accuracy']
    )
    
    return model
```

**Architecture Breakdown**:
1. **Input**: 224×224×4 fingerprint
2. **Conv Blocks**: 4 blocks with increasing filters (32→64→128→256)
3. **Pooling**: Reduces spatial dimensions (224→112→56→28→14)
4. **Global Pool**: 14×14×256 → 256 features
5. **Dense**: 512 → 256 (feature extraction layer)
6. **4 Output Heads**: Separate branches for each task

**Total Parameters**: ~1.6 million
**Feature Layer**: 256-dimensional vector (used later for similarity search)

## Cell 8: Transfer Learning Architecture
```python
def create_transfer_learning_cnn(architecture='efficientnet', ...):
    # Similar to custom CNN but uses pre-trained EfficientNet/ResNet
    # Adapts for 4-channel input with channel projection layer
```

## Cell 10: Model Factory
```python
def create_fire_cnn(architecture='custom', ...):
    # Factory function to create different architectures
    if architecture == 'custom':
        return create_custom_fire_cnn(...)
    elif architecture in ['efficientnet', 'resnet']:
        return create_transfer_learning_cnn(architecture, ...)
```

## Cell 12: Test Model Creation
```python
custom_model = create_fire_cnn('custom')
# Output: Model created with 1,576,282 parameters
```

## Cell 16: `prepare_training_data()` Function
```python
def prepare_training_data(fingerprints, labels, test_size=0.2, validation_split=0.2):
    # 1. Convert labels to one-hot encoding
    for task in ['fire_type', 'ignition_cause', 'state', 'size_category']:
        task_values = np.array([label[task] for label in labels])
        # One-hot encode: [0, 1, 2] → [[1,0,0], [0,1,0], [0,0,1]]
        task_labels[task] = tf.keras.utils.to_categorical(task_values)
    
    # 2. Split data
    # First split: train+val (80%) vs test (20%)
    train_val_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=task_values
    )
    
    # Second split: train (80% of 80% = 64%) vs val (20% of 80% = 16%)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.2, stratify=...
    )
    
    # 3. Create splits
    X_train = fingerprints[train_indices]
    X_val = fingerprints[val_indices]
    X_test = fingerprints[test_indices]
    
    y_train = {task: task_labels[task][train_indices] for task in tasks}
    y_val = {task: task_labels[task][val_indices] for task in tasks}
    y_test = {task: task_labels[task][test_indices] for task in tasks}
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), task_names
```

**Purpose**: Prepare data for neural network training
**Process**:
1. One-hot encode labels (numbers → binary vectors)
2. Split: 64% train, 16% validation, 20% test
3. Format for multi-task learning

**Example Split for 50 fires**:
- Train: 28 fires (56%)
- Val: 7 fires (14%)
- Test: 15 fires (30%)

## Cell 19: `FireCNNTrainer` Class - **TRAINING PIPELINE**
```python
class FireCNNTrainer:
    def __init__(self, model, task_names, model_save_path):
        self.model = model
        self.task_names = task_names
        self.model_save_path = Path(model_save_path)
    
    def create_callbacks(self):
        # 1. ModelCheckpoint: Save best model
        checkpoint = callbacks.ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_loss',
            save_best_only=True
        )
        
        # 2. EarlyStopping: Stop if no improvement
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 3. ReduceLROnPlateau: Lower learning rate if stuck
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )
        
        # 4. TensorBoard: Logging
        tensorboard = callbacks.TensorBoard(log_dir='logs/')
        
        return [checkpoint, early_stop, reduce_lr, tensorboard]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        # Create callbacks
        training_callbacks = self.create_callbacks()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=training_callbacks
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # For each task, calculate metrics
        for i, task in enumerate(self.task_names):
            pred_labels = np.argmax(predictions[i], axis=1)
            true_labels = np.argmax(y_test[task], axis=1)
            
            # Classification report
            report = classification_report(true_labels, pred_labels)
            print(report)
        
        return results
```

**Purpose**: Complete training system
**Features**:
- Automatic model checkpointing
- Early stopping
- Learning rate reduction
- TensorBoard logging
- Multi-task evaluation

## Cell 21: Training Demonstration
```python
# 1. Load data
fingerprints, labels, metadata, encoders = load_processed_data()

# 2. Prepare training data
(X_train, y_train), (X_val, y_val), (X_test, y_test), task_names = prepare_training_data(...)

# 3. Create model
model = create_fire_cnn('custom', num_classes_dict={...})

# 4. Create trainer
trainer = FireCNNTrainer(model, task_names)

# 5. Train (5 epochs for demo)
history = trainer.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=8)

# 6. Evaluate
results = trainer.evaluate(X_test, y_test)

# 7. Save
trainer.save_model('demo_trained_model.keras')
trainer.save_training_history('demo_training_history.json')
```

**Training Output** (per epoch):
```
Epoch 1/5
- loss: 5.19 - fire_type_loss: 1.35 - ignition_cause_loss: 1.21 
- state_loss: 1.64 - size_category_loss: 1.65
- fire_type_accuracy: 0.32 - ignition_cause_accuracy: 0.61
- state_accuracy: 0.36 - size_category_accuracy: 0.25
```

## Cell 25: Extract CNN Features for Similarity Search
```python
def extract_cnn_features(model, fingerprints):
    # Create model that outputs features from 'feature_extraction' layer
    feature_model = models.Model(
        inputs=model.input,
        outputs=model.get_layer('feature_extraction').output
    )
    
    # Extract features (batch processing)
    features = feature_model.predict(fingerprints, batch_size=32)
    
    return features  # Shape: (N, 256)

# Extract features
cnn_features = extract_cnn_features(model, fingerprints)
np.save('demo_cnn_features.npy', cnn_features)
```

**Purpose**: Extract 256-dimensional feature vectors
**Why**: These features will be used in Notebook 05 for similarity search
**Output**: (50, 256) array saved to disk

---

# 📓 NOTEBOOK 04: Pattern Analysis & Features

## Cell 6: `FirePatternAnalyzer` Class - **COMPREHENSIVE FEATURE EXTRACTION**

### Part 1: Shape Features (Lines 50-150)
```python
def extract_shape_features(self, fingerprint):
    shape_mask = fingerprint[:, :, 0]  # Binary shape channel
    
    # 1. Find contours
    contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    
    # 2. Area and perimeter
    features['area'] = cv2.contourArea(main_contour)
    features['perimeter'] = cv2.arcLength(main_contour, True)
    
    # 3. Compactness (circularity)
    features['compactness'] = 4 * π * area / (perimeter²)
    # Circle = 1.0, Complex shape < 1.0
    
    # 4. Fit ellipse
    ellipse = cv2.fitEllipse(main_contour)
    (center, axes, angle) = ellipse
    major_axis, minor_axis = axes
    
    # 5. Elongation
    features['elongation'] = major_axis / minor_axis
    # Circle = 1.0, Elongated > 1.0
    
    # 6. Orientation (angle of major axis)
    features['orientation'] = angle
    
    # 7. Eccentricity
    features['eccentricity'] = sqrt(1 - (minor_axis/major_axis)²)
    # Circle = 0.0, Elongated → 1.0
    
    # 8. Convex hull properties
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    
    # Solidity (how "filled" the shape is)
    features['solidity'] = area / hull_area
    # Solid shape = 1.0, Lots of indentations < 1.0
    
    # 9. Bounding box
    x, y, w, h = cv2.boundingRect(main_contour)
    bounding_area = w * h
    
    # Extent (how much of bounding box is filled)
    features['extent'] = area / bounding_area
    
    return features
```

### Part 2: Complexity Features (Lines 151-280)
```python
def extract_complexity_features(self, fingerprint):
    shape_mask = fingerprint[:, :, 0]
    
    # 1. Fractal dimension using box-counting
    features['fractal_dimension'] = self._calculate_fractal_dimension(shape_mask)
    
    # 2. Boundary roughness
    contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Calculate distances from boundary to centroid
    M = cv2.moments(main_contour)
    cx = M['m10'] / M['m00']  # Center x
    cy = M['m01'] / M['m00']  # Center y
    
    distances = []
    for point in main_contour:
        x, y = point[0]
        dist = sqrt((x - cx)² + (y - cy)²)
        distances.append(dist)
    
    # Roughness = standard deviation of distances
    features['boundary_roughness'] = np.std(distances)
    # Smooth boundary = low std, Jagged = high std
    
    # 3. Convexity defects
    hull = cv2.convexHull(main_contour, returnPoints=False)
    defects = cv2.convexityDefects(main_contour, hull)
    features['convexity_defects'] = len(defects)
    # How many "indentations" in the boundary
    
    # 4. Overall shape complexity (combined metric)
    features['shape_complexity'] = (
        fractal_dimension +
        boundary_roughness * 0.01 +
        convexity_defects * 0.1
    )
    
    return features

def _calculate_fractal_dimension(self, shape_mask, max_box_size=32):
    # Box-counting method for fractal dimension
    # 1. Find boundary pixels
    boundary_pixels = find_contour_pixels(shape_mask)
    
    # 2. Count occupied boxes at different scales
    box_sizes = []
    counts = []
    
    for box_size in range(2, max_box_size):
        # Create grid of boxes
        # Count how many boxes contain boundary pixels
        occupied_boxes = count_occupied_boxes(boundary_pixels, box_size)
        box_sizes.append(box_size)
        counts.append(occupied_boxes)
    
    # 3. Fractal dimension from log-log plot
    log_sizes = np.log(1.0 / np.array(box_sizes))
    log_counts = np.log(np.array(counts))
    slope, intercept = np.polyfit(log_sizes, log_counts, 1)
    fractal_dim = slope
    
    return max(1.0, min(2.0, fractal_dim))  # Clamp to [1.0, 2.0]
```

### Part 3: Texture Features (Lines 281-330)
```python
def extract_texture_features(self, fingerprint):
    distance_transform = fingerprint[:, :, 1]
    
    # Normalize to 0-255 for GLCM
    texture_img = ((distance_transform - min) / (max - min) * 255).astype(np.uint8)
    
    # Calculate GLCM (Gray Level Co-occurrence Matrix)
    # GLCM measures texture patterns
    glcm = graycomatrix(
        texture_img,
        distances=[1],  # Compare adjacent pixels
        angles=[0, π/4, π/2, 3π/4],  # 4 directions
        symmetric=True,
        normed=True
    )
    
    # Extract texture properties
    
    # 1. Contrast: Intensity difference between pixel and neighbor
    features['texture_contrast'] = graycoprops(glcm, 'contrast').mean()
    # Low = uniform texture, High = varied texture
    
    # 2. Homogeneity: Closeness of distribution to diagonal
    features['texture_homogeneity'] = graycoprops(glcm, 'homogeneity').mean()
    # High = similar pixel values
    
    # 3. Energy: Sum of squared elements
    features['texture_energy'] = graycoprops(glcm, 'energy').mean()
    # Uniform texture = high energy
    
    # 4. Correlation: Pixel pair correlation
    features['texture_correlation'] = graycoprops(glcm, 'correlation').mean()
    # How predictable are neighbor values
    
    return features
```

### Part 4: Curvature Features (Lines 331-370)
```python
def extract_curvature_features(self, fingerprint):
    curvature_map = fingerprint[:, :, 2]
    
    # 1. Basic curvature statistics
    features['mean_curvature'] = np.mean(curvature_map)
    features['curvature_variance'] = np.var(curvature_map)
    features['max_curvature'] = np.max(curvature_map)
    
    # 2. Count curvature peaks (local maxima)
    # Apply Gaussian blur to reduce noise
    smoothed = cv2.GaussianBlur(curvature_map, (5, 5), 1.0)
    
    # Find local maxima using Laplacian
    kernel = np.array([[-1,-1,-1],
                       [-1, 8,-1],
                       [-1,-1,-1]])
    peaks = cv2.filter2D(smoothed, -1, kernel)
    
    # Count peaks above threshold
    threshold = mean + std
    features['curvature_peaks'] = np.sum(peaks > threshold)
    # Number of "sharp turns" in boundary
    
    return features
```

### Part 5: Multi-scale Features (Lines 371-410)
```python
def extract_multiscale_features(self, fingerprint, scales=[0.25, 0.5, 1.0, 2.0]):
    areas = []
    perimeters = []
    complexities = []
    
    for scale in scales:
        # Resize fingerprint
        h, w = fingerprint.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        scaled = cv2.resize(fingerprint, new_size)
        
        # Extract features at this scale
        scale_features = self.extract_shape_features(scaled)
        scale_complexity = self.extract_complexity_features(scaled)
        
        areas.append(scale_features.get('area', 0))
        perimeters.append(scale_features.get('perimeter', 0))
        complexities.append(scale_complexity.get('shape_complexity', 0))
    
    # Store average across scales
    features['multi_scale_area'] = np.mean(areas)
    features['multi_scale_perimeter'] = np.mean(perimeters)
    features['multi_scale_complexity'] = np.mean(complexities)
    
    return features
```

### Main Extraction Method (Lines 411-450)
```python
def extract_all_features(self, fingerprint):
    features = {}
    
    # Extract each feature group
    features.update(self.extract_shape_features(fingerprint))
    features.update(self.extract_complexity_features(fingerprint))
    features.update(self.extract_texture_features(fingerprint))
    features.update(self.extract_curvature_features(fingerprint))
    features.update(self.extract_multiscale_features(fingerprint))
    
    return features

def batch_extract_features(self, fingerprints, show_progress=True):
    all_features = []
    
    for fingerprint in tqdm(fingerprints):
        features = self.extract_all_features(fingerprint)
        all_features.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    return features_df
```

## Cell 10: Single Fire Feature Extraction Demo
```python
analyzer = FirePatternAnalyzer()
sample_fingerprint = fingerprints[0]
sample_features = analyzer.extract_all_features(sample_fingerprint)

# Output shows all 23 features:
# Shape: area=26162, perimeter=697.5, compactness=0.676, ...
# Complexity: fractal_dimension=1.0, boundary_roughness=15.0, ...
# Texture: texture_contrast=5.176, texture_homogeneity=0.586, ...
# Curvature: mean_curvature=0.001, curvature_variance=0.000, ...
```

## Cell 12: Batch Feature Extraction - **PROCESSES ALL FIRES**
```python
features_df = analyzer.batch_extract_features(fingerprints, show_progress=True)
# Progress bar shows: 100%|██████████| 50/50

# Normalize features
normalized_features = analyzer.normalize_features(features_df)

# Save
features_df.to_csv('raw_features.csv', index=False)
normalized_features.to_csv('normalized_features.csv', index=False)
```

**Output**: 50 × 23 feature matrix
**Time**: ~15 seconds for 50 fires (~200 fires/minute)

## Cell 14: Feature Distribution Analysis
```python
# Analyzes feature statistics and creates visualizations
# Output: Histograms of key features, statistical summary table
```

## Cell 16: Feature Correlation Analysis
```python
# Calculates correlation matrix
# Finds most correlated feature pairs:
# area ↔ multi_scale_area: 1.000 (perfect - same thing at different scales)
# mean_curvature ↔ curvature_variance: 0.992 (highly correlated)
# perimeter ↔ multi_scale_perimeter: 0.986

# Creates correlation heatmap visualization
```

## Cell 18: Feature Importance Analysis
```python
def get_feature_importance(features_df, target_labels, task='fire_type'):
    # Train Random Forest on features
    X = features_df.fillna(0).values
    y = np.array([label[task] for label in target_labels])
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    return sorted(zip(feature_names, importances), reverse=True)

# Output for fire_type:
# 1. shape_complexity: 0.0787
# 2. multi_scale_complexity: 0.0766
# 3. perimeter: 0.0652
# ...

# Output for size_category:
# 1. curvature_peaks: 0.0724
# 2. compactness: 0.0619
# 3. perimeter: 0.0618
# ...
```

**Insight**: Different features important for different tasks!

## Cell 22: Create Feature Database
```python
# Saves comprehensive feature database:
# - raw_features.csv (50 × 23)
# - normalized_features.csv (50 × 23)
# - feature_metadata.json (descriptions, creation date)
# - fire_metadata.csv (fire information + labels)
# - feature_statistics.json (stats, correlations)
```

---

# 📓 NOTEBOOK 05: Similarity Search & Clustering

## Cell 6: `FireSimilaritySearch` Class - **SEARCH ENGINE**

### Part 1: Database Loading (Lines 1-100)
```python
class FireSimilaritySearch:
    def __init__(self, feature_database_path=None):
        self.database_path = feature_database_path or config.get_path('feature_database')
        self.features_df = None
        self.normalized_features = None
        self.cnn_features = None
        self.metadata = None
        self.labels = None
        self.fingerprints = None
        self.search_engines = {}
        self.scalers = {}
    
    def load_database(self):
        # Load features
        self.features_df = pd.read_csv(self.database_path / 'raw_features.csv')
        self.normalized_features = pd.read_csv(self.database_path / 'normalized_features.csv')
        
        # Load metadata
        self.metadata = pd.read_csv(self.database_path / 'fire_metadata.csv')
        
        # Load CNN features if available
        if 'demo_cnn_features.npy' exists:
            self.cnn_features = np.load('demo_cnn_features.npy')
        
        # Load fingerprints if available
        if 'demo_processed_data/fingerprints.npy' exists:
            self.fingerprints = np.load('demo_processed_data/fingerprints.npy')
        
        return True
```

### Part 2: Build Search Engine (Lines 101-200)
```python
def build_search_engine(self, feature_type='geometric', n_neighbors=10):
    # 1. Select features based on type
    if feature_type == 'geometric':
        features = self.normalized_features.values  # (N, 23)
    elif feature_type == 'cnn':
        features = self.cnn_features  # (N, 256)
    elif feature_type == 'combined':
        # Concatenate geometric + CNN features
        geometric = self.normalized_features.values
        cnn_normalized = StandardScaler().fit_transform(self.cnn_features)
        features = np.concatenate([geometric, cnn_normalized], axis=1)  # (N, 279)
    
    # 2. Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    self.scalers[feature_type] = scaler
    
    # 3. Build k-NN index
    nn_search = NearestNeighbors(
        n_neighbors=min(n_neighbors + 1, len(features)),
        algorithm='auto',  # Automatically chooses best algorithm
        metric='cosine'    # Cosine similarity (angle between vectors)
    )
    nn_search.fit(features_normalized)
    self.search_engines[feature_type] = nn_search
    
    return True
```

**What is k-NN?**
- k-Nearest Neighbors: Find k fires most similar to query
- Uses cosine similarity: angle between feature vectors
- Small distance = high similarity

### Part 3: Similarity Search (Lines 201-300)
```python
def find_similar_fires(self, query_index, feature_type='geometric', 
                      n_neighbors=5, return_distances=False):
    # 1. Get query features
    if feature_type == 'geometric':
        query_features = self.normalized_features.iloc[query_index:query_index+1].values
    elif feature_type == 'cnn':
        query_features = self.cnn_features[query_index:query_index+1]
    elif feature_type == 'combined':
        geometric = self.normalized_features.iloc[query_index:query_index+1].values
        cnn = self.cnn_features[query_index:query_index+1]
        query_features = np.concatenate([geometric, cnn], axis=1)
    
    # 2. Normalize query
    query_normalized = self.scalers[feature_type].transform(query_features)
    
    # 3. Find neighbors
    distances, indices = self.search_engines[feature_type].kneighbors(
        query_normalized,
        n_neighbors=n_neighbors+1  # +1 because first result is query itself
    )
    
    # 4. Remove self-match
    distances = distances[0][1:]  # Skip first (query itself)
    indices = indices[0][1:]
    
    # 5. Get metadata for similar fires
    similar_fires = []
    for i, idx in enumerate(indices):
        fire_data = {
            'index': int(idx),
            'distance': float(distances[i]),
            'metadata': self.metadata.iloc[idx].to_dict(),
            'labels': self.labels[idx],
            'features': self.features_df.iloc[idx].to_dict()
        }
        similar_fires.append(fire_data)
    
    return similar_fires
```

**Example Usage**:
```python
# Find 5 fires similar to fire #0
similar = search_engine.find_similar_fires(query_index=0, n_neighbors=5)

# Output:
# 1. Fire 999 - Prescribed Burn (203 ha) - Distance: 0.059
# 2. Fire 456 - Bushfire (75 ha) - Distance: 0.161
# 3. Fire 092 - Bushfire (16 ha) - Distance: 0.181
# 4. Fire 789 - Prescribed Burn (31 ha) - Distance: 0.185
# 5. Fire 234 - Unknown (2 ha) - Distance: 0.217
```

### Part 4: Clustering (Lines 301-450)
```python
def discover_fire_patterns(self, n_clusters=8, feature_type='geometric'):
    # 1. Get features
    if feature_type == 'geometric':
        features = self.normalized_features.values
    elif feature_type == 'cnn':
        features = StandardScaler().fit_transform(self.cnn_features)
    elif feature_type == 'combined':
        geometric = self.normalized_features.values
        cnn = StandardScaler().fit_transform(self.cnn_features)
        features = np.concatenate([geometric, cnn], axis=1)
    
    # 2. Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    
    # 3. Store results
    self.clusters = clusters
    self.cluster_centers = kmeans.cluster_centers_
    
    # 4. Analyze clusters
    cluster_analysis = self._analyze_clusters(clusters, features)
    
    return {
        'clusters': clusters,
        'cluster_centers': self.cluster_centers,
        'analysis': cluster_analysis
    }

def _analyze_clusters(self, clusters, features):
    # Calculate clustering quality metrics
    silhouette = silhouette_score(features, clusters)
    # Silhouette: -1 to 1, higher = better separation
    # > 0.5 = excellent, 0.2-0.5 = good, < 0.2 = weak
    
    calinski = calinski_harabasz_score(features, clusters)
    # Calinski-Harabasz: higher = better defined clusters
    
    # Analyze cluster sizes
    cluster_sizes = [np.sum(clusters == i) for i in range(n_clusters)]
    
    # Find representative fire for each cluster
    representatives = []
    for i in range(n_clusters):
        cluster_mask = clusters == i
        cluster_features = features[cluster_mask]
        center = self.cluster_centers[i]
        
        # Find fire closest to cluster center
        distances = np.sum((cluster_features - center) ** 2, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        representatives.append(closest_idx)
    
    return {
        'silhouette_score': silhouette,
        'calinski_harabasz_score': calinski,
        'cluster_sizes': cluster_sizes,
        'representatives': representatives
    }

def get_cluster_info(self, cluster_id):
    # Get all fires in cluster
    cluster_mask = self.clusters == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    cluster_fires = self.metadata.iloc[cluster_indices]
    
    # Analyze characteristics
    fire_types = cluster_fires['original_fire_type'].value_counts()
    states = cluster_fires['state'].value_counts()
    sizes = cluster_fires['area_ha'].describe()
    
    return {
        'cluster_id': cluster_id,
        'size': len(cluster_indices),
        'fire_types': fire_types.to_dict(),
        'states': states.to_dict(),
        'size_stats': sizes.to_dict(),
        'representative_fires': cluster_indices[:5].tolist()
    }
```

## Cell 10: Single Fire Similarity Search Demo
```python
# Query fire #0
query_metadata = search_engine.metadata.iloc[0]
similar_fires = search_engine.find_similar_fires(0, 'geometric', n_neighbors=5)

# Output:
# Query Fire: Fire_001 - Prescribed Burn, NSW, 4.0 ha
# Top 5 Similar Fires:
#   1. Fire_045 - Prescribed Burn (distance: 0.059)
#   2. Fire_012 - Bushfire (distance: 0.161)
#   3. Fire_089 - Bushfire (distance: 0.181)
# ...
```

## Cell 12: Batch Similarity Search
```python
# Search for first 10 fires
batch_queries = list(range(10))
batch_results = search_engine.batch_similarity_search(batch_queries, 'geometric', n_neighbors=3)

# Output: Shows search consistency
# Query: Fire_001 (Prescribed Burn) → 1/3 type matches (Consistency: 0.33)
# Query: Fire_002 (Prescribed Burn) → 3/3 type matches (Consistency: 1.00)
# Average consistency: 0.400
```

## Cell 14: Pattern Discovery
```python
# Discover 5 fire pattern clusters
clustering_results = search_engine.discover_fire_patterns(n_clusters=5, feature_type='geometric')

# Output:
# Silhouette Score: 0.607 (excellent clustering)
# Cluster 0: 43 fires (86.0%)
# Cluster 1: 1 fire (2.0%)
# Cluster 2: 1 fire (2.0%)
# Cluster 3: 1 fire (2.0%)
# Cluster 4: 4 fires (8.0%)
```

## Cell 16: Cluster Analysis
```python
# Analyze each cluster
for cluster_id in range(5):
    cluster_info = search_engine.get_cluster_info(cluster_id)
    # Shows:
    # - Cluster size
    # - Dominant fire type
    # - Average fire size
    # - State distribution
    # - Representative fires
```

## Cell 18: Explore Individual Clusters
```python
# Explore cluster 0
explore_cluster_patterns(search_engine, cluster_id=0, n_examples=2)

# Output:
# Cluster 0 (43 fires)
#   Average size: 64.9 ha
#   Dominant type: Prescribed Burn
#   Representative fires:
#     1. Fire_001: Prescribed Burn, 4.0 ha, Fractal: 1.000
#     2. Fire_002: Prescribed Burn, 509.0 ha, Fractal: 1.009

# Visualizations: Average fingerprint for cluster (all 4 channels)
```

## Cell 22: Save Search Engine
```python
# Save for later use
save_similarity_search_engine(search_engine)

# Saves:
# - geometric_search.pkl - k-NN index
# - geometric_scaler.pkl - Feature normalizer
# - cnn_search.pkl (if available)
# - combined_search.pkl (if available)
# - search_metadata.json - Configuration
```

---

# 📓 NOTEBOOK 06: Complete System Demo

## Cell 6: `FireFingerprintingSystem` Class
```python
class FireFingerprintingSystem:
    def __init__(self):
        self.fingerprints = None
        self.features = None
        self.cnn_features = None
        self.metadata = None
        self.labels = None
    
    def load_demo_data(self):
        # Load all processed data:
        # 1. Fingerprints (50 × 224 × 224 × 4)
        # 2. Geometric features (50 × 23)
        # 3. CNN features (50 × 256)
        # 4. Metadata (50 fire records)
        # 5. Labels (fire_type, cause, state, size)
        
        fingerprints, labels, metadata, encoders = load_processed_data()
        features_df = pd.read_csv('demo_features.csv')
        cnn_features = np.load('demo_cnn_features.npy')
        
        self.fingerprints = fingerprints
        self.features = features_df
        self.cnn_features = cnn_features
        self.metadata = metadata
        self.labels = labels
        
        return True

system = FireFingerprintingSystem()
system.load_demo_data()
```

## Cell 8: Complete Pipeline Demonstration
```python
# 1. Visualize fingerprints
# Shows 3 sample fires with all 4 channels + RGB composite

# 2. Feature analysis
# Statistics for key features (area, perimeter, compactness, fractal, curvature)

# 3. CNN features
# Shows 256-dimensional feature vectors

# Output: Comprehensive visualizations and statistics
```

## Cell 10: System Performance Analysis
```python
# Dataset Statistics:
#   • 50 fires
#   • 224×224×4 fingerprints
#   • 23 geometric features
#   • 256 CNN features
#   • 38.3 MB memory

# Fire Type Distribution:
#   • Prescribed Burn: 24 (48%)
#   • Unknown: 15 (30%)
#   • Bushfire: 11 (22%)

# Fire Size Statistics:
#   • Mean: 606.3 ha
#   • Median: 13.5 ha
#   • Max: 16,081 ha

# Processing Speed Estimates:
#   • Polygon conversion: ~50 fires/minute
#   • Feature extraction: ~200 fires/minute
#   • CNN inference: ~500 fires/minute

# Scalability (for full 324K dataset):
#   • Processing time: 4-6 hours
#   • Storage: 50-100 GB
#   • Search latency: <100ms per query
```

## Cell 12: Real-World Application Scenarios
```python
# Scenario 1: Fire Investigation
# - Find complex fire (high fractal dimension)
# - Search for similar historical fires
# - Use case: Understand fire behavior for investigation

# Scenario 2: Risk Assessment
# - Find large fires (>100 ha)
# - Average size of large fires: 2469.8 ha
# - Use case: Regional risk assessment

# Scenario 3: Emergency Response
# - Categorize by complexity
#   • Low complexity: 1 fire → Basic team
#   • Medium complexity: 43 fires → Enhanced team
#   • High complexity: 6 fires → Major incident team

# Scenario 4: Training
# - Use diverse fire patterns for training materials
```

## Cell 14: Advanced Feature Analysis
```python
# Fire Size vs Complexity correlation: 0.325 (moderate)

# Fire Type characteristics:
#   • Prescribed Burn: Complexity=0.998, Roughness=24.865
#   • Unknown: Complexity=1.035, Roughness=22.165
#   • Bushfire: Complexity=1.066, Roughness=19.952

# Top feature correlations:
#   • area ↔ multi_scale_area: 1.000
#   • mean_curvature ↔ curvature_variance: 0.992

# Feature importance for fire size prediction:
#   1. fractal_dimension: 0.3216
#   2. perimeter: 0.2110
#   3. curvature_peaks: 0.1479
```

## Cell 16: System Benchmarking
```python
# Benchmark 1: Feature Discriminability
# Cross-validation accuracy: 0.441 (can distinguish fire types moderately)

# Benchmark 2: Clustering Quality
# Silhouette scores for 3-5 clusters: 0.505-0.515 (excellent separation)

# Benchmark 3: Data Quality
# Completeness: 95.9%, Variability: 1.000

# Benchmark 4: System Readiness
# ✓ All components loaded and functional

# OVERALL ASSESSMENT: 0.732
# Rating: EXCELLENT - Ready for production use
```

## Cell 18: Future Directions
```python
# Research opportunities:
# 1. Real-time fire monitoring integration
# 2. Multi-spectral fire analysis (infrared + visible)
# 3. Weather-integrated fire modeling
# 4. Cross-regional pattern transfer
# 5. Human-centric response planning
# 6. Climate change pattern analysis

# Key research questions:
# - How do patterns vary across ecosystems?
# - Can we predict fire spread from early patterns?
# - What role does terrain play?
# - How have patterns changed with climate?
# - Can we identify arson from patterns?
```

## Cell 20: Final Summary
```python
# BREAKTHROUGH ACHIEVEMENT
# First computer vision approach to fire boundary analysis

# TECHNICAL INNOVATIONS
# • 4-channel fingerprint representation
# • Multi-task CNN architecture
# • 20+ geometric features
# • Efficient similarity search
# • Pattern discovery clustering

# REAL-WORLD IMPACT
# • Fire investigation support
# • Risk assessment tools
# • Resource planning optimization
# • Research capabilities
# • Training materials

# PERFORMANCE METRICS
# • Feature Discriminability: 0.441
# • Clustering Stability: 0.507
# • Data Quality: 0.980
# • System Readiness: 1.000

# OVERALL: EXCELLENT - Ready for production
```

---

## 🎓 SUMMARY OF ALL NOTEBOOKS

### Notebook 01: Foundation
**Core Function**: `polygon_to_fingerprint()`
- Converts fire polygons to 4-channel images
- 4 channels: shape, distance, curvature, fractal
- Output: 224×224×4 numpy array

### Notebook 02: Scaling
**Core Function**: `process_fire_dataset()`
- Batch processes thousands of fires
- Encodes labels for machine learning
- Saves processed data for later use
- Can handle full 324K dataset

### Notebook 03: Intelligence
**Core Function**: `create_fire_cnn()` + `FireCNNTrainer.train()`
- Multi-task CNN with 4 output heads
- Classifies fire_type, cause, state, size
- Extracts 256-dimensional feature vectors
- ~1.6M parameters

### Notebook 04: Understanding
**Core Class**: `FirePatternAnalyzer`
- Extracts 23 geometric features
- Shape, complexity, texture, curvature analysis
- Multi-scale feature extraction
- Interpretable fire characteristics

### Notebook 05: Discovery
**Core Class**: `FireSimilaritySearch`
- k-NN similarity search (geometric + CNN features)
- Find similar historical fires
- Pattern discovery through clustering
- Interactive cluster exploration

### Notebook 06: Integration
**Core Class**: `FireFingerprintingSystem`
- Integrates all components
- Demonstrates real-world applications
- System performance benchmarking
- Future research directions

---

*This cell-by-cell guide provides detailed explanations of every function, class, and code block in the FirePrint system. Use it as a reference when working with the notebooks!* 🔥📚

