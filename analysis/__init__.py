# analysis/__init__.py
from .utils import (
    load_parquet, load_embeddings, prepare_data, compute_pca_rgb,
    compute_features, align_arrays, compute_correlations
)

# UPDATED: Import the correct names found in plotting.py [cite: 8, 24, 32, 39, 47]
from .plotting import (
    plot_embeddings_rgb,            # Was plot_embeddings_on_map
    plot_oriented_embeddings_rgb,   # Was plot_oriented_embeddings
    plot_cluster_on_map,            # Was plot_random_cluster
    plot_correlations,
    plot_cluster_lidars,
    comprehensive_analysis
)

from .map_utils import get_map_image_path

__all__ = [
    'load_parquet', 'load_embeddings', 'prepare_data', 'compute_pca_rgb',
    'compute_features', 'align_arrays', 'compute_correlations',
    'plot_embeddings_rgb', 'plot_oriented_embeddings_rgb',
    'plot_cluster_on_map', 'plot_correlations', 'plot_cluster_lidars',
    'comprehensive_analysis',
    'get_map_image_path'
]