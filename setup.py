#!/usr/bin/env python
"""
FirePrint v1.0 Setup Script
Computer Vision System for Wildfire Pattern Analysis
"""

from setuptools import setup, find_packages
import os

# Read version from version.yaml
def read_version():
    """Read version from version.yaml"""
    import yaml
    with open('version.yaml', 'r') as f:
        version_info = yaml.safe_load(f)
    return version_info['version']

# Read long description from README
def read_long_description():
    """Read long description from README.md"""
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt"""
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='fireprint',
    version=read_version(),
    author='FirePrint Team',
    author_email='your.email@example.com',
    description='Computer Vision System for Wildfire Boundary Pattern Analysis',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/FirePrint-v1.0',
    project_urls={
        'Bug Tracker': 'https://github.com/yourusername/FirePrint-v1.0/issues',
        'Documentation': 'https://github.com/yourusername/FirePrint-v1.0/blob/main/docs/DOCUMENTATION.md',
        'Source Code': 'https://github.com/yourusername/FirePrint-v1.0',
    },
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'flake8>=6.0.0',
            'black>=23.0.0',
            'mypy>=1.0.0',
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinx-autodoc-typehints>=1.0.0',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'wildfire',
        'computer-vision',
        'deep-learning',
        'cnn',
        'geospatial',
        'pattern-recognition',
        'fire-science',
        'similarity-search',
        'feature-extraction',
    ],
)

