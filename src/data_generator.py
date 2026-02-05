"""
Synthetic Test Data Generator
Generates test data matching training statistics with injected anomalies
"""

import numpy as np
import pandas as pd
from scipy import stats


class SyntheticDataGenerator:
    """
    Generates synthetic test data for predictive maintenance testing
    """
    
    def __init__(self, training_data, time_column='Time'):
        """
        Initialize generator with training data statistics
        
        Args:
            training_data (pd.DataFrame): Training data
            time_column (str): Name of time column
        """
        self.training_data = training_data
        self.time_column = time_column
        self.statistics = self._calculate_statistics()
        
    def _calculate_statistics(self):
        """
        Calculate statistics from training data
        
        Returns:
            dict: Statistics for each axis
        """
        stats_dict = {}
        
        for col in self.training_data.columns:
            if col.startswith('Axis_'):
                stats_dict[col] = {
                    'mean': self.training_data[col].mean(),
                    'std': self.training_data[col].std(),
                    'min': self.training_data[col].min(),
                    'max': self.training_data[col].max(),
                    'median': self.training_data[col].median(),
                    'q25': self.training_data[col].quantile(0.25),
                    'q75': self.training_data[col].quantile(0.75)
                }
        
        # Time statistics
        stats_dict[self.time_column] = {
            'min': self.training_data[self.time_column].min(),
            'max': self.training_data[self.time_column].max(),
            'range': self.training_data[self.time_column].max() - self.training_data[self.time_column].min()
        }
        
        return stats_dict
    
    def generate_test_data(self, n_samples=1000, anomaly_rate=0.10, 
                          extend_time=True, random_seed=42):
        """
        Generate synthetic test data
        
        Args:
            n_samples (int): Number of samples to generate
            anomaly_rate (float): Proportion of anomalous samples (0-1)
            extend_time (bool): Whether to extend time beyond training range
            random_seed (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Synthetic test data
        """
        np.random.seed(random_seed)
        
        synthetic_data = pd.DataFrame()
        
        # Generate time column
        time_stats = self.statistics[self.time_column]
        if extend_time:
            # Extend time by 50% beyond training range
            time_start = time_stats['max']
            time_end = time_stats['max'] + time_stats['range'] * 0.5
        else:
            time_start = time_stats['min']
            time_end = time_stats['max']
        
        synthetic_data[self.time_column] = np.linspace(time_start, time_end, n_samples)
        
        # Generate data for each axis
        n_normal = int(n_samples * (1 - anomaly_rate))
        n_anomalies = n_samples - n_normal
        
        for axis_col, axis_stats in self.statistics.items():
            if not axis_col.startswith('Axis_'):
                continue
            
            mean = axis_stats['mean']
            std = axis_stats['std']
            
            # Generate normal data
            normal_data = np.random.normal(mean, std, n_normal)
            
            # Generate anomalies (2-3 std deviations above mean)
            anomaly_mean = mean + np.random.uniform(2, 3) * std
            anomaly_std = std * 0.5  # Tighter distribution for anomalies
            anomalies = np.random.normal(anomaly_mean, anomaly_std, n_anomalies)
            
            # Combine and shuffle
            all_values = np.concatenate([normal_data, anomalies])
            np.random.shuffle(all_values)
            
            synthetic_data[axis_col] = all_values
        
        print(f"✅ Generated {n_samples} synthetic samples")
        print(f"   Normal samples: {n_normal} ({(1-anomaly_rate)*100:.1f}%)")
        print(f"   Anomaly samples: {n_anomalies} ({anomaly_rate*100:.1f}%)")
        print(f"   Time range: {synthetic_data[self.time_column].min():.2f} - {synthetic_data[self.time_column].max():.2f}")
        
        return synthetic_data
    
    def print_statistics_comparison(self, synthetic_data):
        """
        Compare statistics between training and synthetic data
        
        Args:
            synthetic_data (pd.DataFrame): Generated synthetic data
        """
        print("\n📊 Statistics Comparison:\n")
        print(f"{'Axis':<10} {'Train Mean':<12} {'Test Mean':<12} {'Train Std':<12} {'Test Std':<12}")
        print("-" * 60)
        
        for axis_col in self.statistics.keys():
            if not axis_col.startswith('Axis_'):
                continue
            
            train_mean = self.statistics[axis_col]['mean']
            train_std = self.statistics[axis_col]['std']
            test_mean = synthetic_data[axis_col].mean()
            test_std = synthetic_data[axis_col].std()
            
            print(f"{axis_col:<10} {train_mean:<12.4f} {test_mean:<12.4f} {train_std:<12.4f} {test_std:<12.4f}")
    
    def save_statistics(self, filepath='data/processed/training_statistics.csv'):
        """
        Save training statistics to CSV
        
        Args:
            filepath (str): Output file path
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        stats_df = pd.DataFrame(self.statistics).T
        stats_df.to_csv(filepath)
        print(f"💾 Statistics saved to {filepath}")


def generate_advanced_synthetic_data(training_data, models, n_samples=1000,
                                     anomaly_rate=0.10, random_seed=42):
    """
    Generate synthetic data using trained regression models
    This creates more realistic data that follows the learned patterns
    
    Args:
        training_data (pd.DataFrame): Original training data
        models (dict): Dictionary of trained regression models
        n_samples (int): Number of samples to generate
        anomaly_rate (float): Proportion of anomalous samples
        random_seed (int): Random seed
        
    Returns:
        pd.DataFrame: Synthetic test data
    """
    np.random.seed(random_seed)
    
    synthetic_data = pd.DataFrame()
    
    # Generate extended time range
    time_min = training_data['Time'].min()
    time_max = training_data['Time'].max()
    time_range = time_max - time_min
    
    # Extend time by 50%
    new_time_max = time_max + time_range * 0.5
    synthetic_data['Time'] = np.linspace(time_max, new_time_max, n_samples)
    
    # Generate data for each axis using regression models
    for axis_num, model in models.items():
        axis_col = f'Axis_{axis_num}'
        
        # Get predictions from model
        predictions = model.predict(synthetic_data['Time'])
        
        # Add normal noise based on training residuals
        time_train = training_data['Time']
        current_train = training_data[axis_col]
        residuals = model.calculate_residuals(time_train, current_train)
        residual_std = np.std(residuals)
        
        # Create normal samples
        n_normal = int(n_samples * (1 - anomaly_rate))
        n_anomalies = n_samples - n_normal
        
        # Normal data: predictions + small noise
        normal_noise = np.random.normal(0, residual_std, n_normal)
        
        # Anomalous data: predictions + large positive noise
        anomaly_noise = np.random.normal(2.5 * residual_std, residual_std * 0.5, n_anomalies)
        
        # Combine noises
        all_noise = np.concatenate([normal_noise, anomaly_noise])
        np.random.shuffle(all_noise)
        
        # Final values
        synthetic_data[axis_col] = predictions + all_noise
    
    print(f"✅ Generated {n_samples} model-based synthetic samples")
    return synthetic_data


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing SyntheticDataGenerator...\n")
    
    # Create sample training data
    np.random.seed(42)
    n_train = 500
    training_data = pd.DataFrame({
        'Time': np.linspace(0, 1000, n_train),
        'Axis_1': np.random.normal(50, 5, n_train),
        'Axis_2': np.random.normal(45, 4, n_train),
        'Axis_3': np.random.normal(55, 6, n_train)
    })
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(training_data)
    synthetic_data = generator.generate_test_data(n_samples=200, anomaly_rate=0.15)
    
    # Compare statistics
    generator.print_statistics_comparison(synthetic_data)
    
    print("\n✅ Test completed successfully!")
