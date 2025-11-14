# analysis/map_utils.py
"""
Utilities for finding map images from simulation metadata.
"""
import os
import json

def get_map_image_path(data_output_dir):
    """
    Get map image path from metadata.json in the data output directory.
    
    Args:
        data_output_dir: Directory containing simulation data (e.g., "output/2025-11-14-213949_random_walk_10k")
    
    Returns:
        str: Path to map image file, or None if not found
    """
    if not os.path.exists(data_output_dir):
        print(f"[WARNING] Directory not found: {data_output_dir}")
        return None
    
    # Look for metadata.json
    metadata_path = os.path.join(data_output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"[WARNING] metadata.json not found in: {data_output_dir}")
        return None
    
    # Read map image from metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        map_image_name = metadata.get("environment_parameters", {}).get("map_image")
        if not map_image_name:
            print("[WARNING] No map_image found in metadata")
            return None
        
        # The map image should be in environments/images relative to the project root
        map_image_path = os.path.join("environments", "images", map_image_name)
        
        if os.path.exists(map_image_path):
            return map_image_path
        else:
            print(f"[WARNING] Map image not found at: {map_image_path}")
            return None
            
    except Exception as e:
        print(f"[WARNING] Error reading metadata: {e}")
        return None