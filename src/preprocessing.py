"""
Data preprocessing utilities
Includes normalization and standardization functions
"""

import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    Handles normalization and standardization of data
    """
    
    def __init__(self):
        self.normalization_params = {}
        self.standardization_params = {}
        
    def fit_normalization(self, data, columns=None):
        """
        Calculate min-max normalization parameters from training data
        
        Args:
            data (pd.DataFrame): Training data
            columns (list): Columns to normalize (None = all numeric)
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            self.normalization_params[col] = {
                'min': data[col].min(),
                'max': data[col].max()
            }
        
        print(f"✅ Normalization parameters fitted for {len(columns)} columns")
    
    def normalize(self, data, columns=None):
        """
        Apply min-max normalization: (x - min) / (max - min)
        Scales values to [0, 1] range
        
        Args:
            data (pd.DataFrame): Data to normalize
            columns (list): Columns to normalize (None = all fitted)
            
        Returns:
            pd.DataFrame: Normalized data
        """
        if not self.normalization_params:
            raise ValueError("Must call fit_normalization() first")
        
        if columns is None:
            columns = list(self.normalization_params.keys())
        
        normalized_data = data.copy()
        
        for col in columns:
            if col not in self.normalization_params:
                print(f"⚠️  Warning: No normalization params for {col}, skipping")
                continue
            
            min_val = self.normalization_params[col]['min']
            max_val = self.normalization_params[col]['max']
            
            # Avoid division by zero
            if max_val - min_val == 0:
                normalized_data[col] = 0
            else:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
        
        return normalized_data
    
    def denormalize(self, data, columns=None):
        """
        Reverse min-max normalization
        
        Args:
            data (pd.DataFrame): Normalized data
            columns (list): Columns to denormalize
            
        Returns:
            pd.DataFrame: Original scale data
        """
        if columns is None:
            columns = list(self.normalization_params.keys())
        
        denormalized_data = data.copy()
        
        for col in columns:
            if col not in self.normalization_params:
                continue
            
            min_val = self.normalization_params[col]['min']
            max_val = self.normalization_params[col]['max']
            
            denormalized_data[col] = data[col] * (max_val - min_val) + min_val
        
        return denormalized_data
    
    def fit_standardization(self, data, columns=None):
        """
        Calculate z-score standardization parameters from training data
        
        Args:
            data (pd.DataFrame): Training data
            columns (list): Columns to standardize (None = all numeric)
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            self.standardization_params[col] = {
                'mean': data[col].mean(),
                'std': data[col].std()
            }
        
        print(f"✅ Standardization parameters fitted for {len(columns)} columns")
    
    def standardize(self, data, columns=None):
        """
        Apply z-score standardization: (x - mean) / std
        Centers data around 0 with standard deviation of 1
        
        Args:
            data (pd.DataFrame): Data to standardize
            columns (list): Columns to standardize (None = all fitted)
            
        Returns:
            pd.DataFrame: Standardized data
        """
        if not self.standardization_params:
            raise ValueError("Must call fit_standardization() first")
        
        if columns is None:
            columns = list(self.standardization_params.keys())
        
        standardized_data = data.copy()
        
        for col in columns:
            if col not in self.standardization_params:
                print(f"⚠️  Warning: No standardization params for {col}, skipping")
                continue
            
            mean_val = self.standardization_params[col]['mean']
            std_val = self.standardization_params[col]['std']
            
            # Avoid division by zero
            if std_val == 0:
                standardized_data[col] = 0
            else:
                standardized_data[col] = (data[col] - mean_val) / std_val
        
        return standardized_data
    
    def destandardize(self, data, columns=None):
        """
        Reverse z-score standardization
        
        Args:
            data (pd.DataFrame): Standardized data
            columns (list): Columns to destandardize
            
        Returns:
            pd.DataFrame: Original scale data
        """
        if columns is None:
            columns = list(self.standardization_params.keys())
        
        destandardized_data = data.copy()
        
        for col in columns:
            if col not in self.standardization_params:
                continue
            
            mean_val = self.standardization_params[col]['mean']
            std_val = self.standardization_params[col]['std']
            
            destandardized_data[col] = data[col] * std_val + mean_val
        
        return destandardized_data
    
    def get_params_summary(self):
        """
        Get summary of fitted parameters
        
        Returns:
            dict: Summary of normalization and standardization params
        """
        summary = {
            'normalization': {},
            'standardization': {}
        }
        
        for col, params in self.normalization_params.items():
            summary['normalization'][col] = {
                'min': params['min'],
                'max': params['max'],
                'range': params['max'] - params['min']
            }
        
        for col, params in self.standardization_params.items():
            summary['standardization'][col] = {
                'mean': params['mean'],
                'std': params['std']
            }
        
        return summary
    
    def save_params(self, filepath):
        """
        Save preprocessing parameters to file
        
        Args:
            filepath (str): Output file path
        """
        import pickle
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        params = {
            'normalization': self.normalization_params,
            'standardization': self.standardization_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        
        print(f"💾 Preprocessing parameters saved to {filepath}")
    
    def load_params(self, filepath):
        """
        Load preprocessing parameters from file
        
        Args:
            filepath (str): Input file path
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        self.normalization_params = params['normalization']
        self.standardization_params = params['standardization']
        
        print(f"📂 Preprocessing parameters loaded from {filepath}")


def calculate_z_scores(data, column):
    """
    Calculate z-scores for outlier detection
    
    Args:
        data (pd.Series or np.array): Data to analyze
        column (str): Column name for reporting
        
    Returns:
        tuple: (z_scores, outlier_indices)
    """
    mean = np.mean(data)
    std = np.std(data)
    
    z_scores = (data - mean) / std
    
    # Outliers are typically defined as |z| > 3
    outlier_indices = np.where(np.abs(z_scores) > 3)[0]
    
    print(f"📊 {column} Z-Score Analysis:")
    print(f"   Mean: {mean:.4f}")
    print(f"   Std: {std:.4f}")
    print(f"   Outliers (|z| > 3): {len(outlier_indices)} ({len(outlier_indices)/len(data)*100:.2f}%)")
    
    return z_scores, outlier_indices


def detect_outliers_iqr(data, column, multiplier=1.5):
    """
    Detect outliers using Interquartile Range (IQR) method
    
    Args:
        data (pd.Series or np.array): Data to analyze
        column (str): Column name for reporting
        multiplier (float): IQR multiplier (typically 1.5)
        
    Returns:
        tuple: (lower_bound, upper_bound, outlier_indices)
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
    
    print(f"📊 {column} IQR Analysis:")
    print(f"   Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
    print(f"   Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"   Outliers: {len(outlier_indices)} ({len(outlier_indices)/len(data)*100:.2f}%)")
    
    return lower_bound, upper_bound, outlier_indices


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing DataPreprocessor...\n")
    
    # Create sample data
    np.random.seed(42)
    train_data = pd.DataFrame({
        'Time': np.linspace(0, 100, 100),
        'Axis_1': np.random.normal(50, 10, 100),
        'Axis_2': np.random.normal(45, 8, 100)
    })
    
    test_data = pd.DataFrame({
        'Time': np.linspace(100, 150, 50),
        'Axis_1': np.random.normal(52, 10, 50),
        'Axis_2': np.random.normal(46, 8, 50)
    })
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Fit and apply normalization
    print("1. Min-Max Normalization:")
    preprocessor.fit_normalization(train_data)
    normalized_test = preprocessor.normalize(test_data)
    print(f"   Original range: [{test_data['Axis_1'].min():.2f}, {test_data['Axis_1'].max():.2f}]")
    print(f"   Normalized range: [{normalized_test['Axis_1'].min():.2f}, {normalized_test['Axis_1'].max():.2f}]")
    
    # Fit and apply standardization
    print("\n2. Z-Score Standardization:")
    preprocessor.fit_standardization(train_data)
    standardized_test = preprocessor.standardize(test_data)
    print(f"   Original mean/std: {test_data['Axis_1'].mean():.2f} / {test_data['Axis_1'].std():.2f}")
    print(f"   Standardized mean/std: {standardized_test['Axis_1'].mean():.2f} / {standardized_test['Axis_1'].std():.2f}")
    
    # Outlier detection
    print("\n3. Outlier Detection:")
    calculate_z_scores(train_data['Axis_1'], 'Axis_1')
    
    print("\n✅ Test completed successfully!")
