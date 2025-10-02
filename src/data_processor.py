#!/usr/bin/env python3
"""
Fire Dataset Processor
Processes the entire Australian bushfire dataset and converts to fingerprints
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from polygon_converter import polygon_to_fingerprint, batch_convert_polygons

class FireDataProcessor:
    """Process fire dataset and convert to fingerprints"""
    
    def __init__(self, gdb_path, output_dir="processed_data"):
        self.gdb_path = gdb_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Label encoders
        self.fire_type_encoder = {}
        self.cause_encoder = {}
        self.state_encoder = {}
        self.size_encoder = {}
        
    def load_fire_data(self, layer_name="Bushfire_Boundaries_Historical_V3"):
        """Load fire data from geodatabase"""
        print(f"Loading fire data from {self.gdb_path}...")
        
        try:
            gdf = gpd.read_file(self.gdb_path, layer=layer_name)
            print(f"Loaded {len(gdf):,} fire records")
            
            # Basic data info
            print(f"Columns: {list(gdf.columns)}")
            print(f"Geometry types: {gdf.geometry.type.value_counts()}")
            
            return gdf
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def analyze_labels(self, gdf):
        """Analyze label distributions for encoding"""
        print("\n" + "="*60)
        print("LABEL ANALYSIS")
        print("="*60)
        
        # Fire types
        fire_types = gdf['fire_type'].value_counts()
        print(f"\nFire Types ({fire_types.sum():,} records):")
        for fire_type, count in fire_types.items():
            print(f"  {fire_type}: {count:,} ({count/len(gdf)*100:.1f}%)")
        
        # Ignition causes
        cause_data = gdf[gdf['ignition_cause'].notna()]
        if len(cause_data) > 0:
            causes = cause_data['ignition_cause'].value_counts()
            print(f"\nIgnition Causes ({len(cause_data):,} records with data):")
            for cause, count in causes.head(10).items():
                print(f"  {cause}: {count:,} ({count/len(cause_data)*100:.1f}%)")
        
        # States
        states = gdf['state'].value_counts()
        print(f"\nStates/Territories ({states.sum():,} records):")
        for state, count in states.items():
            print(f"  {state}: {count:,} ({count/len(gdf)*100:.1f}%)")
        
        # Fire sizes
        area_stats = gdf['area_ha'].describe()
        print(f"\nFire Size Statistics:")
        print(f"  Mean: {area_stats['mean']:.1f} ha")
        print(f"  Median: {area_stats['50%']:.1f} ha")
        print(f"  Max: {area_stats['max']:,.0f} ha")
        print(f"  Min: {area_stats['min']:.1f} ha")
        
        return {
            'fire_types': fire_types,
            'causes': causes if len(cause_data) > 0 else pd.Series(),
            'states': states,
            'area_stats': area_stats
        }
    
    def create_label_encoders(self, gdf):
        """Create label encoders for categorical variables"""
        print("\nCreating label encoders...")
        
        # Fire type encoder
        fire_types = gdf['fire_type'].dropna().unique()
        self.fire_type_encoder = {ft: i for i, ft in enumerate(sorted(fire_types))}
        
        # Ignition cause encoder (top causes only)
        cause_data = gdf[gdf['ignition_cause'].notna()]
        if len(cause_data) > 0:
            top_causes = cause_data['ignition_cause'].value_counts().head(10).index
            self.cause_encoder = {cause: i for i, cause in enumerate(top_causes)}
            self.cause_encoder['Other'] = len(top_causes)  # For rare causes
        
        # State encoder
        states = gdf['state'].dropna().unique()
        self.state_encoder = {state: i for i, state in enumerate(sorted(states))}
        
        # Size category encoder (based on common fire size classifications)
        self.size_encoder = {
            'Small': 0,      # < 10 ha
            'Medium': 1,     # 10-100 ha
            'Large': 2,      # 100-1000 ha
            'Very Large': 3  # > 1000 ha
        }
        
        print(f"Fire types: {len(self.fire_type_encoder)} categories")
        print(f"Ignition causes: {len(self.cause_encoder)} categories")
        print(f"States: {len(self.state_encoder)} categories")
        print(f"Size categories: {len(self.size_encoder)} categories")
        
        # Save encoders
        encoders = {
            'fire_type': self.fire_type_encoder,
            'ignition_cause': self.cause_encoder,
            'state': self.state_encoder,
            'size_category': self.size_encoder
        }
        
        with open(self.output_dir / 'label_encoders.json', 'w') as f:
            json.dump(encoders, f, indent=2)
        
        return encoders
    
    def encode_fire_type(self, fire_type):
        """Encode fire type"""
        if pd.isna(fire_type) or fire_type not in self.fire_type_encoder:
            return 0  # Default to first category
        return self.fire_type_encoder[fire_type]
    
    def encode_ignition_cause(self, cause):
        """Encode ignition cause"""
        if pd.isna(cause):
            return len(self.cause_encoder) - 1  # 'Other' category
        if cause in self.cause_encoder:
            return self.cause_encoder[cause]
        else:
            return self.cause_encoder['Other']
    
    def encode_state(self, state):
        """Encode state"""
        if pd.isna(state) or state not in self.state_encoder:
            return 0  # Default to first state
        return self.state_encoder[state]
    
    def encode_size_category(self, area_ha):
        """Encode fire size category"""
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
    
    def filter_valid_geometries(self, gdf):
        """Filter out invalid or problematic geometries"""
        print("\nFiltering valid geometries...")
        
        initial_count = len(gdf)
        
        # Remove null geometries
        gdf = gdf[gdf.geometry.notna()]
        print(f"After removing null geometries: {len(gdf):,} ({len(gdf)/initial_count*100:.1f}%)")
        
        # Remove invalid geometries
        valid_mask = gdf.geometry.is_valid
        gdf = gdf[valid_mask]
        print(f"After removing invalid geometries: {len(gdf):,} ({len(gdf)/initial_count*100:.1f}%)")
        
        # Remove very small geometries (< 0.1 ha)
        area_mask = gdf['area_ha'] >= 0.1
        gdf = gdf[area_mask]
        print(f"After removing tiny fires (<0.1 ha): {len(gdf):,} ({len(gdf)/initial_count*100:.1f}%)")
        
        # Remove geometries with zero area bounds
        def has_valid_bounds(geom):
            try:
                bounds = geom.bounds
                return (bounds[2] - bounds[0]) > 0 and (bounds[3] - bounds[1]) > 0
            except:
                return False
        
        bounds_mask = gdf.geometry.apply(has_valid_bounds)
        gdf = gdf[bounds_mask]
        print(f"After removing zero-area bounds: {len(gdf):,} ({len(gdf)/initial_count*100:.1f}%)")
        
        return gdf.reset_index(drop=True)
    
    def process_fire_dataset(self, sample_size=None, image_size=224, batch_size=1000):
        """Process entire fire dataset to fingerprints"""
        print("="*60)
        print("PROCESSING FIRE DATASET TO FINGERPRINTS")
        print("="*60)
        
        # Load data
        gdf = self.load_fire_data()
        if gdf is None:
            return None, None, None
        
        # Analyze labels
        label_stats = self.analyze_labels(gdf)
        
        # Create encoders
        encoders = self.create_label_encoders(gdf)
        
        # Filter valid geometries
        gdf = self.filter_valid_geometries(gdf)
        
        # Sample if requested
        if sample_size and sample_size < len(gdf):
            print(f"\nSampling {sample_size:,} fires from {len(gdf):,} total...")
            gdf = gdf.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        print(f"\nProcessing {len(gdf):,} fires to fingerprints...")
        
        # Process in batches
        all_fingerprints = []
        all_labels = []
        all_metadata = []
        failed_count = 0
        
        for batch_start in tqdm(range(0, len(gdf), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(gdf))
            batch_gdf = gdf.iloc[batch_start:batch_end]
            
            # Convert geometries to fingerprints
            batch_fingerprints = []
            batch_labels = []
            batch_metadata = []
            
            for idx, fire in batch_gdf.iterrows():
                try:
                    # Convert to fingerprint
                    fingerprint = polygon_to_fingerprint(fire.geometry, image_size)
                    
                    if fingerprint is not None:
                        batch_fingerprints.append(fingerprint)
                        
                        # Prepare labels
                        labels = {
                            'fire_type': self.encode_fire_type(fire.fire_type),
                            'ignition_cause': self.encode_ignition_cause(fire.ignition_cause),
                            'state': self.encode_state(fire.state),
                            'size_category': self.encode_size_category(fire.area_ha)
                        }
                        batch_labels.append(labels)
                        
                        # Store metadata
                        metadata = {
                            'fire_id': fire.fire_id if 'fire_id' in fire else idx,
                            'area_ha': fire.area_ha,
                            'ignition_date': str(fire.ignition_date) if pd.notna(fire.ignition_date) else None,
                            'original_fire_type': fire.fire_type,
                            'original_cause': fire.ignition_cause,
                            'original_state': fire.state
                        }
                        batch_metadata.append(metadata)
                    else:
                        failed_count += 1
                
                except Exception as e:
                    failed_count += 1
                    continue
            
            # Add batch results
            if batch_fingerprints:
                all_fingerprints.extend(batch_fingerprints)
                all_labels.extend(batch_labels)
                all_metadata.extend(batch_metadata)
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {len(all_fingerprints):,} fires")
        print(f"Failed conversions: {failed_count:,}")
        print(f"Success rate: {len(all_fingerprints)/(len(all_fingerprints)+failed_count)*100:.1f}%")
        
        # Convert to numpy arrays
        fingerprints = np.array(all_fingerprints)
        
        # Save processed data
        self.save_processed_data(fingerprints, all_labels, all_metadata, label_stats)
        
        return fingerprints, all_labels, all_metadata
    
    def save_processed_data(self, fingerprints, labels, metadata, label_stats):
        """Save processed data to disk"""
        print(f"\nSaving processed data to {self.output_dir}...")
        
        # Save fingerprints (in chunks if large)
        if len(fingerprints) > 10000:
            # Save in chunks for memory efficiency
            chunk_size = 5000
            for i in range(0, len(fingerprints), chunk_size):
                chunk_end = min(i + chunk_size, len(fingerprints))
                chunk_fingerprints = fingerprints[i:chunk_end]
                
                np.save(
                    self.output_dir / f'fingerprints_chunk_{i//chunk_size:03d}.npy',
                    chunk_fingerprints
                )
        else:
            np.save(self.output_dir / 'fingerprints.npy', fingerprints)
        
        # Save labels and metadata
        with open(self.output_dir / 'labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
        
        with open(self.output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save statistics
        stats = {
            'total_fingerprints': len(fingerprints),
            'fingerprint_shape': fingerprints.shape,
            'label_stats': {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                           for k, v in label_stats.items()},
            'processing_date': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'processing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print("Data saved successfully!")
        print(f"Files created:")
        print(f"  - fingerprints.npy (or chunks)")
        print(f"  - labels.pkl")
        print(f"  - metadata.pkl")
        print(f"  - label_encoders.json")
        print(f"  - processing_stats.json")
    
    def load_processed_data(self):
        """Load previously processed data"""
        print(f"Loading processed data from {self.output_dir}...")
        
        try:
            # Load fingerprints
            fingerprint_file = self.output_dir / 'fingerprints.npy'
            if fingerprint_file.exists():
                fingerprints = np.load(fingerprint_file)
            else:
                # Load from chunks
                chunk_files = sorted(self.output_dir.glob('fingerprints_chunk_*.npy'))
                if chunk_files:
                    fingerprint_chunks = [np.load(f) for f in chunk_files]
                    fingerprints = np.concatenate(fingerprint_chunks, axis=0)
                else:
                    raise FileNotFoundError("No fingerprint files found")
            
            # Load labels and metadata
            with open(self.output_dir / 'labels.pkl', 'rb') as f:
                labels = pickle.load(f)
            
            with open(self.output_dir / 'metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            # Load encoders
            with open(self.output_dir / 'label_encoders.json', 'r') as f:
                encoders = json.load(f)
            
            print(f"Loaded {len(fingerprints):,} fingerprints")
            print(f"Fingerprint shape: {fingerprints.shape}")
            
            return fingerprints, labels, metadata, encoders
        
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None, None, None, None

def main():
    """Main processing function"""
    # Set up processor
    gdb_path = "../data/Bushfire_Boundaries_Historical_2024_V3.gdb"
    processor = FireDataProcessor(gdb_path)
    
    # Process sample dataset first (for testing)
    print("Processing sample dataset (1000 fires) for testing...")
    fingerprints, labels, metadata = processor.process_fire_dataset(sample_size=1000)
    
    if fingerprints is not None:
        print(f"\nSample processing successful!")
        print(f"Generated {len(fingerprints):,} fingerprints")
        print(f"Fingerprint shape: {fingerprints.shape}")
        
        # Show some statistics
        print(f"\nFingerprint statistics:")
        for i in range(4):
            channel = fingerprints[:, :, :, i]
            print(f"  Channel {i+1}: mean={channel.mean():.3f}, std={channel.std():.3f}")

if __name__ == "__main__":
    main()
