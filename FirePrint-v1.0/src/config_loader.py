"""
FirePrint Configuration Loader
================================
Utility to load and access configuration from config.yaml

Usage in notebooks:
    from src.config_loader import FirePrintConfig
    
    config = FirePrintConfig()
    
    # Access paths
    gdb_path = config.get_path('source_data.bushfire_gdb')
    demo_data_dir = config.get_path('processed_data.demo')
    
    # Access parameters
    image_size = config.get('processing.image_size')
    batch_size = config.get('model.batch_size')
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import os


class FirePrintConfig:
    """Load and manage FirePrint configuration"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration loader
        
        Args:
            config_file: Path to config.yaml. If None, searches for it automatically.
        """
        if config_file is None:
            # Try to find config.yaml in common locations
            possible_paths = [
                Path(__file__).parent.parent / 'config.yaml',  # FirePrint-v1.0/config.yaml
                Path.cwd() / 'config.yaml',  # Current directory
                Path.cwd() / 'FirePrint-v1.0' / 'config.yaml',  # One level up
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_file = str(path)
                    break
            
            if config_file is None:
                raise FileNotFoundError(
                    "Could not find config.yaml. Please specify the path explicitly."
                )
        
        self.config_file = Path(config_file)
        self.config_dir = self.config_file.parent
        
        # Load configuration
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key in dot notation (e.g., 'processing.image_size')
            default: Default value if key not found
        
        Returns:
            Configuration value
        
        Example:
            config.get('model.batch_size')  # Returns 32
            config.get('processing.image_size')  # Returns 224
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_path(self, key: str, resolve: bool = True, create: bool = False) -> Path:
        """
        Get path from configuration and resolve it relative to config directory
        
        Args:
            key: Path key in dot notation (e.g., 'processed_data.demo')
            resolve: If True, resolve relative to config directory
            create: If True, create directory if it doesn't exist
        
        Returns:
            Path object
        
        Example:
            config.get_path('paths.processed_data.demo')
            config.get_path('paths.models.demo_training', create=True)
        """
        # Get the path value
        if not key.startswith('paths.'):
            key = f'paths.{key}'
        
        path_str = self.get(key)
        
        if path_str is None:
            raise KeyError(f"Path key '{key}' not found in configuration")
        
        path = Path(path_str)
        
        # Resolve relative paths
        if resolve and not path.is_absolute():
            path = (self.config_dir / path).resolve()
        
        # Create directory if requested
        if create and not path.exists():
            if '.' in path.name:  # It's a file path
                path.parent.mkdir(parents=True, exist_ok=True)
            else:  # It's a directory path
                path.mkdir(parents=True, exist_ok=True)
        
        return path
    
    def get_file_path(self, directory_key: str, file_key: str, 
                      create_dir: bool = False) -> Path:
        """
        Get complete file path by combining directory and filename
        
        Args:
            directory_key: Key for directory path (e.g., 'processed_data.demo')
            file_key: Key for filename (e.g., 'fingerprints')
            create_dir: If True, create directory if it doesn't exist
        
        Returns:
            Complete file path
        
        Example:
            config.get_file_path('processed_data.demo', 'fingerprints')
            # Returns: Path('../data/demo_processed_data/fingerprints.npy')
        """
        directory = self.get_path(directory_key, create=create_dir)
        
        if not file_key.startswith('files.'):
            file_key = f'files.{file_key}'
        
        filename = self.get(file_key)
        
        if filename is None:
            raise KeyError(f"File key '{file_key}' not found in configuration")
        
        return directory / filename
    
    def get_all_paths(self) -> Dict[str, Path]:
        """
        Get all configured paths as a dictionary
        
        Returns:
            Dictionary of all paths with their keys
        """
        paths = {}
        
        def extract_paths(d: dict, prefix: str = ''):
            for key, value in d.items():
                current_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    extract_paths(value, current_key)
                elif isinstance(value, str):
                    paths[current_key] = self.get_path(f'paths.{current_key}')
        
        if 'paths' in self.config:
            extract_paths(self.config['paths'])
        
        return paths
    
    def update_path(self, key: str, new_path: str):
        """
        Update a path in the configuration and save to file
        
        Args:
            key: Path key (e.g., 'source_data.bushfire_gdb')
            new_path: New path value
        """
        if not key.startswith('paths.'):
            key = f'paths.{key}'
        
        keys = key.split('.')
        current = self.config
        
        # Navigate to the correct location
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Update the value
        current[keys[-1]] = new_path
        
        # Save updated config
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def print_config(self, section: Optional[str] = None):
        """
        Print configuration in a readable format
        
        Args:
            section: If specified, print only this section (e.g., 'paths', 'model')
        """
        import json
        
        if section:
            data = self.get(section, {})
        else:
            data = self.config
        
        print(json.dumps(data, indent=2, default=str))
    
    def __repr__(self):
        return f"FirePrintConfig(config_file='{self.config_file}')"


# Convenience function for quick access
def load_config(config_file: Optional[str] = None) -> FirePrintConfig:
    """
    Load FirePrint configuration
    
    Args:
        config_file: Path to config.yaml (optional)
    
    Returns:
        FirePrintConfig instance
    """
    return FirePrintConfig(config_file)


# Example usage
if __name__ == "__main__":
    # Load config
    config = load_config()
    
    print("=" * 60)
    print("FirePrint Configuration Loaded")
    print("=" * 60)
    print(f"Config file: {config.config_file}")
    print(f"Project: {config.get('project.name')} v{config.get('project.version')}")
    print()
    
    # Print paths
    print("Configured Paths:")
    print("-" * 60)
    print(f"GDB Path: {config.get_path('source_data.bushfire_gdb')}")
    print(f"Demo Data: {config.get_path('processed_data.demo')}")
    print(f"Models: {config.get_path('models.demo_training')}")
    print(f"Outputs: {config.get_path('outputs_root')}")
    print()
    
    # Print parameters
    print("Processing Parameters:")
    print("-" * 60)
    print(f"Image Size: {config.get('processing.image_size')}")
    print(f"Batch Size: {config.get('processing.batch_size')}")
    print(f"Model Architecture: {config.get('model.architecture')}")
    print()
    
    print("=" * 60)

