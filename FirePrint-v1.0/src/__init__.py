"""
FirePrint v1.0 - Computer Vision System for Wildfire Pattern Analysis

This package provides tools for converting fire polygon boundaries into
4-channel fingerprint images and analyzing them using deep learning.

Main modules:
    - polygon_converter: Convert fire polygons to fingerprints
    - data_processor: Process and prepare datasets
    - cnn_model: Build and configure CNN architectures
    - feature_extractor: Extract geometric and textural features
    - similarity_search: Search for similar fire patterns
    - trainer: Train and evaluate models

Example:
    >>> from fireprint import polygon_to_fingerprint, FireCNNTrainer
    >>> fingerprint = polygon_to_fingerprint(fire_polygon)
    >>> # Train model on fingerprints...
"""

__version__ = '1.0.0'
__author__ = 'FirePrint Team'
__license__ = 'MIT'

# Note: Actual implementation modules to be created from notebooks
# For now, users should use the Jupyter notebooks directly

__all__ = [
    '__version__',
    '__author__',
    '__license__',
]

