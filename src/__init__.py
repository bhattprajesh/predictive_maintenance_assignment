"""
Predictive Maintenance Package
"""

from .database import connect_to_database, fetch_training_data
from .regression import AxisRegressionModel, MultiAxisRegressionSystem
from .data_generator import SyntheticDataGenerator, generate_advanced_synthetic_data
from .preprocessing import DataPreprocessor
from .anomaly_detection import AnomalyDetector, discover_thresholds_from_residuals
from .visualization import (
    plot_regression_with_alerts,
    plot_residual_analysis,
    plot_all_axes_comparison,
    plot_anomaly_summary,
    create_results_summary_report
)

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    # Database
    'connect_to_database',
    'fetch_training_data',
    
    # Regression
    'AxisRegressionModel',
    'MultiAxisRegressionSystem',
    
    # Data Generation
    'SyntheticDataGenerator',
    'generate_advanced_synthetic_data',
    
    # Preprocessing
    'DataPreprocessor',
    
    # Anomaly Detection
    'AnomalyDetector',
    'discover_thresholds_from_residuals',
    
    # Visualization
    'plot_regression_with_alerts',
    'plot_residual_analysis',
    'plot_all_axes_comparison',
    'plot_anomaly_summary',
    'create_results_summary_report'
]
