#!/usr/bin/env python3
"""
Fire Pattern Analyzer
Extract geometric and textural features from fire fingerprints for analysis
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from skimage import measure, morphology
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class FirePatternAnalyzer:
    """Analyze fire patterns and extract features from fingerprints"""
    
    def __init__(self):
        self.feature_names = []
        self.feature_descriptions = {}
        self._initialize_feature_definitions()
    
    def _initialize_feature_definitions(self):
        """Initialize feature names and descriptions"""
        self.feature_descriptions = {
            # Shape-based features
            'area': 'Total area of fire boundary',
            'perimeter': 'Total perimeter length',
            'compactness': 'Compactness ratio (4π*area/perimeter²)',
            'elongation': 'Elongation ratio (major/minor axis)',
            'solidity': 'Ratio of area to convex hull area',
            'extent': 'Ratio of area to bounding box area',
            'eccentricity': 'Eccentricity of fitted ellipse',
            'orientation': 'Orientation of major axis',
            
            # Complexity features
            'fractal_dimension': 'Fractal dimension of boundary',
            'boundary_roughness': 'Roughness of fire boundary',
            'convexity_defects': 'Number of convexity defects',
            'shape_complexity': 'Overall shape complexity measure',
            
            # Texture features (from distance transform)
            'texture_contrast': 'Contrast in distance transform',
            'texture_homogeneity': 'Homogeneity in distance transform',
            'texture_energy': 'Energy in distance transform',
            'texture_correlation': 'Correlation in distance transform',
            
            # Curvature features
            'mean_curvature': 'Mean boundary curvature',
            'curvature_variance': 'Variance in boundary curvature',
            'max_curvature': 'Maximum boundary curvature',
            'curvature_peaks': 'Number of curvature peaks',
            
            # Multi-scale features
            'multi_scale_area': 'Area at different scales',
            'multi_scale_perimeter': 'Perimeter at different scales',
            'scale_invariant_features': 'Scale-invariant shape descriptors'
        }
    
    def extract_shape_features(self, shape_mask):
        """Extract shape-based features from binary mask"""
        features = {}
        
        try:
            # Basic shape properties
            features['area'] = np.sum(shape_mask)
            
            # Find contours
            contours, _ = cv2.findContours(
                shape_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) == 0:
                return {key: 0.0 for key in self.feature_descriptions.keys() if 'shape' in key or key in ['area', 'perimeter', 'compactness', 'elongation', 'solidity', 'extent', 'eccentricity', 'orientation']}
            
            # Use largest contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Perimeter
            features['perimeter'] = cv2.arcLength(main_contour, True)
            
            # Compactness
            if features['perimeter'] > 0:
                features['compactness'] = 4 * np.pi * features['area'] / (features['perimeter'] ** 2)
            else:
                features['compactness'] = 0.0
            
            # Fit ellipse for elongation and other properties
            if len(main_contour) >= 5:
                ellipse = cv2.fitEllipse(main_contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                
                features['elongation'] = major_axis / minor_axis if minor_axis > 0 else 1.0
                features['orientation'] = ellipse[2]  # Angle in degrees
                features['eccentricity'] = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0.0
            else:
                features['elongation'] = 1.0
                features['orientation'] = 0.0
                features['eccentricity'] = 0.0
            
            # Convex hull properties
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = features['area'] / hull_area if hull_area > 0 else 0.0
            
            # Bounding box properties
            x, y, w, h = cv2.boundingRect(main_contour)
            bbox_area = w * h
            features['extent'] = features['area'] / bbox_area if bbox_area > 0 else 0.0
            
        except Exception as e:
            print(f"Error extracting shape features: {e}")
            # Return default values
            default_features = ['area', 'perimeter', 'compactness', 'elongation', 'solidity', 'extent', 'eccentricity', 'orientation']
            features = {key: 0.0 for key in default_features}
        
        return features
    
    def extract_complexity_features(self, shape_mask, curvature_map):
        """Extract complexity and fractal features"""
        features = {}
        
        try:
            # Fractal dimension using box-counting
            features['fractal_dimension'] = self._calculate_fractal_dimension(shape_mask)
            
            # Boundary roughness
            features['boundary_roughness'] = self._calculate_boundary_roughness(shape_mask)
            
            # Convexity defects
            features['convexity_defects'] = self._count_convexity_defects(shape_mask)
            
            # Shape complexity (combination of multiple measures)
            features['shape_complexity'] = self._calculate_shape_complexity(shape_mask)
            
        except Exception as e:
            print(f"Error extracting complexity features: {e}")
            features = {
                'fractal_dimension': 1.0,
                'boundary_roughness': 0.0,
                'convexity_defects': 0,
                'shape_complexity': 0.0
            }
        
        return features
    
    def extract_texture_features(self, distance_map):
        """Extract texture features from distance transform"""
        features = {}
        
        try:
            # Normalize distance map to 0-255 for texture analysis
            if distance_map.max() > distance_map.min():
                normalized = ((distance_map - distance_map.min()) / 
                             (distance_map.max() - distance_map.min()) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(distance_map, dtype=np.uint8)
            
            # Gray-Level Co-occurrence Matrix (GLCM) features
            distances = [1, 2, 3]
            angles = [0, 45, 90, 135]
            
            glcm = graycomatrix(
                normalized, 
                distances=distances, 
                angles=np.radians(angles),
                levels=256,
                symmetric=True,
                normed=True
            )
            
            # Calculate GLCM properties
            features['texture_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
            features['texture_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
            features['texture_energy'] = np.mean(graycoprops(glcm, 'energy'))
            features['texture_correlation'] = np.mean(graycoprops(glcm, 'correlation'))
            
        except Exception as e:
            print(f"Error extracting texture features: {e}")
            features = {
                'texture_contrast': 0.0,
                'texture_homogeneity': 1.0,
                'texture_energy': 1.0,
                'texture_correlation': 0.0
            }
        
        return features
    
    def extract_curvature_features(self, curvature_map):
        """Extract curvature-based features"""
        features = {}
        
        try:
            # Only consider non-zero curvature values
            curvature_values = curvature_map[curvature_map > 0]
            
            if len(curvature_values) > 0:
                features['mean_curvature'] = np.mean(curvature_values)
                features['curvature_variance'] = np.var(curvature_values)
                features['max_curvature'] = np.max(curvature_values)
                
                # Count curvature peaks (local maxima)
                features['curvature_peaks'] = self._count_curvature_peaks(curvature_map)
            else:
                features['mean_curvature'] = 0.0
                features['curvature_variance'] = 0.0
                features['max_curvature'] = 0.0
                features['curvature_peaks'] = 0
                
        except Exception as e:
            print(f"Error extracting curvature features: {e}")
            features = {
                'mean_curvature': 0.0,
                'curvature_variance': 0.0,
                'max_curvature': 0.0,
                'curvature_peaks': 0
            }
        
        return features
    
    def extract_all_features(self, fingerprint):
        """Extract all features from a 4-channel fingerprint"""
        if fingerprint.shape[-1] != 4:
            raise ValueError("Fingerprint must have 4 channels")
        
        # Extract individual channels
        shape_mask = fingerprint[:, :, 0]
        distance_map = fingerprint[:, :, 1]
        curvature_map = fingerprint[:, :, 2]
        fractal_map = fingerprint[:, :, 3]
        
        # Extract features from each component
        features = {}
        
        # Shape features
        shape_features = self.extract_shape_features(shape_mask)
        features.update(shape_features)
        
        # Complexity features
        complexity_features = self.extract_complexity_features(shape_mask, curvature_map)
        features.update(complexity_features)
        
        # Texture features
        texture_features = self.extract_texture_features(distance_map)
        features.update(texture_features)
        
        # Curvature features
        curvature_features = self.extract_curvature_features(curvature_map)
        features.update(curvature_features)
        
        return features
    
    def _calculate_fractal_dimension(self, binary_image):
        """Calculate fractal dimension using box-counting method"""
        try:
            # Find boundary pixels
            boundary = cv2.Canny(binary_image.astype(np.uint8) * 255, 50, 150)
            
            if np.sum(boundary) == 0:
                return 1.0
            
            # Box-counting method
            sizes = np.logspace(0.5, 3, num=10, dtype=int)
            counts = []
            
            for size in sizes:
                if size >= min(boundary.shape):
                    continue
                    
                # Count boxes containing boundary pixels
                count = 0
                for i in range(0, boundary.shape[0], size):
                    for j in range(0, boundary.shape[1], size):
                        box = boundary[i:i+size, j:j+size]
                        if np.any(box):
                            count += 1
                counts.append(count)
            
            if len(counts) < 2:
                return 1.0
            
            # Fit line to log-log plot
            sizes = sizes[:len(counts)]
            coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
            fractal_dim = -coeffs[0]
            
            # Clamp to reasonable range
            return max(1.0, min(2.0, fractal_dim))
            
        except Exception:
            return 1.0
    
    def _calculate_boundary_roughness(self, binary_image):
        """Calculate boundary roughness"""
        try:
            contours, _ = cv2.findContours(
                binary_image.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) == 0:
                return 0.0
            
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate roughness as ratio of actual perimeter to convex hull perimeter
            actual_perimeter = cv2.arcLength(main_contour, True)
            hull = cv2.convexHull(main_contour)
            hull_perimeter = cv2.arcLength(hull, True)
            
            if hull_perimeter > 0:
                return actual_perimeter / hull_perimeter - 1.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _count_convexity_defects(self, binary_image):
        """Count convexity defects in shape"""
        try:
            contours, _ = cv2.findContours(
                binary_image.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) == 0:
                return 0
            
            main_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(main_contour, returnPoints=False)
            
            if len(hull) < 4:
                return 0
            
            defects = cv2.convexityDefects(main_contour, hull)
            
            if defects is not None:
                # Count significant defects (depth > threshold)
                significant_defects = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    if d > 1000:  # Threshold for significant defect
                        significant_defects += 1
                return significant_defects
            else:
                return 0
                
        except Exception:
            return 0
    
    def _calculate_shape_complexity(self, binary_image):
        """Calculate overall shape complexity measure"""
        try:
            # Combine multiple complexity measures
            area = np.sum(binary_image)
            
            if area == 0:
                return 0.0
            
            # Perimeter-to-area ratio
            contours, _ = cv2.findContours(
                binary_image.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) == 0:
                return 0.0
            
            perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)
            pa_ratio = perimeter / np.sqrt(area) if area > 0 else 0
            
            # Normalize to 0-1 range
            complexity = min(1.0, pa_ratio / 10.0)
            
            return complexity
            
        except Exception:
            return 0.0
    
    def _count_curvature_peaks(self, curvature_map):
        """Count peaks in curvature map"""
        try:
            # Find local maxima in curvature
            from scipy.ndimage import maximum_filter
            
            # Apply maximum filter to find local maxima
            local_maxima = maximum_filter(curvature_map, size=3) == curvature_map
            
            # Only consider significant peaks
            threshold = np.percentile(curvature_map[curvature_map > 0], 75) if np.any(curvature_map > 0) else 0
            significant_peaks = local_maxima & (curvature_map > threshold)
            
            return np.sum(significant_peaks)
            
        except Exception:
            return 0
    
    def analyze_fingerprint_batch(self, fingerprints, save_results=True, output_dir="analysis_results"):
        """Analyze a batch of fingerprints and extract features"""
        
        if save_results:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        
        print(f"Analyzing {len(fingerprints)} fingerprints...")
        
        all_features = []
        feature_names = None
        
        for i, fingerprint in enumerate(fingerprints):
            try:
                features = self.extract_all_features(fingerprint)
                
                if feature_names is None:
                    feature_names = list(features.keys())
                
                all_features.append(list(features.values()))
                
            except Exception as e:
                print(f"Error analyzing fingerprint {i}: {e}")
                # Add default values
                if feature_names is not None:
                    all_features.append([0.0] * len(feature_names))
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features, columns=feature_names)
        
        if save_results:
            # Save features
            features_df.to_csv(output_path / 'extracted_features.csv', index=False)
            
            # Save feature descriptions
            with open(output_path / 'feature_descriptions.json', 'w') as f:
                json.dump(self.feature_descriptions, f, indent=2)
            
            # Create feature analysis plots
            self._create_feature_analysis_plots(features_df, output_path)
        
        print(f"Feature extraction completed. Shape: {features_df.shape}")
        
        return features_df
    
    def _create_feature_analysis_plots(self, features_df, output_path):
        """Create analysis plots for extracted features"""
        
        # Feature distribution plots
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, column in enumerate(features_df.columns[:16]):  # Plot first 16 features
            if i < len(axes):
                features_df[column].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{column}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(features_df.columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation matrix
        plt.figure(figsize=(12, 10))
        correlation_matrix = features_df.corr()
        sns.heatmap(
            correlation_matrix, 
            annot=False, 
            cmap='coolwarm', 
            center=0,
            square=True
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_path / 'feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plots saved to {output_path}")

def test_pattern_analyzer():
    """Test the pattern analyzer with sample data"""
    print("Testing Fire Pattern Analyzer...")
    
    # Create sample fingerprint
    sample_fingerprint = np.random.random((224, 224, 4))
    
    # Create analyzer
    analyzer = FirePatternAnalyzer()
    
    # Extract features
    features = analyzer.extract_all_features(sample_fingerprint)
    
    print(f"Extracted {len(features)} features:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
    
    print("Pattern analyzer test completed!")

if __name__ == "__main__":
    test_pattern_analyzer()
