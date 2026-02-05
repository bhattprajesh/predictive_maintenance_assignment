"""
Linear Regression Models for Predictive Maintenance
One model per robot axis (8 total)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import os


class AxisRegressionModel:
    """
    Linear regression model for a single robot axis
    Predicts current consumption based on time
    """
    
    def __init__(self, axis_number):
        """
        Initialize model for specific axis
        
        Args:
            axis_number (int): Axis number (1-8)
        """
        self.axis_number = axis_number
        self.model = LinearRegression()
        self.slope = None
        self.intercept = None
        self.r2_score = None
        self.rmse = None
        self.mae = None
        self.is_trained = False
        
    def fit(self, time, current):
        """
        Train the linear regression model
        
        Args:
            time (array-like): Time values (seconds)
            current (array-like): Current consumption values (kWh)
        """
        # Reshape for sklearn
        X = np.array(time).reshape(-1, 1)
        y = np.array(current)
        
        # Fit model
        self.model.fit(X, y)
        
        # Store parameters
        self.slope = self.model.coef_[0]
        self.intercept = self.model.intercept_
        
        # Calculate metrics
        predictions = self.model.predict(X)
        self.r2_score = r2_score(y, predictions)
        self.rmse = np.sqrt(mean_squared_error(y, predictions))
        self.mae = mean_absolute_error(y, predictions)
        
        self.is_trained = True
        
        print(f"✅ Axis {self.axis_number} trained:")
        print(f"   Slope: {self.slope:.6f}")
        print(f"   Intercept: {self.intercept:.6f}")
        print(f"   R² Score: {self.r2_score:.4f}")
        print(f"   RMSE: {self.rmse:.4f}")
        
    def predict(self, time):
        """
        Make predictions for given time values
        
        Args:
            time (array-like): Time values
            
        Returns:
            np.array: Predicted current values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        X = np.array(time).reshape(-1, 1)
        return self.model.predict(X)
    
    def calculate_residuals(self, time, current):
        """
        Calculate residuals (actual - predicted)
        
        Args:
            time (array-like): Time values
            current (array-like): Actual current values
            
        Returns:
            np.array: Residual values
        """
        predictions = self.predict(time)
        return np.array(current) - predictions
    
    def save_model(self, filepath):
        """
        Save model to file
        
        Args:
            filepath (str): Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load model from file
        
        Args:
            filepath (str): Path to model file
            
        Returns:
            AxisRegressionModel: Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"📂 Model loaded from {filepath}")
        return model
    
    def get_statistics(self):
        """
        Get model statistics as dictionary
        
        Returns:
            dict: Model statistics
        """
        return {
            'axis': self.axis_number,
            'slope': self.slope,
            'intercept': self.intercept,
            'r2_score': self.r2_score,
            'rmse': self.rmse,
            'mae': self.mae
        }


class MultiAxisRegressionSystem:
    """
    Manages regression models for all 8 axes
    """
    
    def __init__(self):
        self.models = {}
        
    def train_all_axes(self, df, time_column='Time'):
        """
        Train models for all 8 axes
        
        Args:
            df (pd.DataFrame): Training data
            time_column (str): Name of time column
        """
        print("🚀 Training regression models for all axes...\n")
        
        for i in range(1, 9):
            axis_col = f'Axis_{i}'
            
            if axis_col not in df.columns:
                print(f"⚠️  Warning: {axis_col} not found in data")
                continue
            
            # Create and train model
            model = AxisRegressionModel(i)
            model.fit(df[time_column], df[axis_col])
            
            self.models[i] = model
            print()
        
        print("✅ All models trained successfully!\n")
    
    def predict_all_axes(self, time):
        """
        Make predictions for all axes
        
        Args:
            time (array-like): Time values
            
        Returns:
            dict: Predictions for each axis
        """
        predictions = {}
        for axis_num, model in self.models.items():
            predictions[axis_num] = model.predict(time)
        return predictions
    
    def save_all_models(self, directory='models'):
        """
        Save all models to directory
        
        Args:
            directory (str): Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        for axis_num, model in self.models.items():
            filepath = os.path.join(directory, f'axis_{axis_num}_model.pkl')
            model.save_model(filepath)
    
    def load_all_models(self, directory='models'):
        """
        Load all models from directory
        
        Args:
            directory (str): Directory containing models
        """
        self.models = {}
        for i in range(1, 9):
            filepath = os.path.join(directory, f'axis_{i}_model.pkl')
            if os.path.exists(filepath):
                self.models[i] = AxisRegressionModel.load_model(filepath)
    
    def get_all_statistics(self):
        """
        Get statistics for all models
        
        Returns:
            list: List of statistics dictionaries
        """
        return [model.get_statistics() for model in self.models.values()]


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing AxisRegressionModel...")
    
    # Generate sample data
    np.random.seed(42)
    time = np.linspace(0, 100, 100)
    current = 2.5 * time + 10 + np.random.normal(0, 5, 100)
    
    # Train model
    model = AxisRegressionModel(1)
    model.fit(time, current)
    
    # Make predictions
    predictions = model.predict(time[:10])
    print(f"\n📊 Sample predictions: {predictions[:5]}")
    
    # Calculate residuals
    residuals = model.calculate_residuals(time, current)
    print(f"📊 Residual mean: {np.mean(residuals):.4f}")
    print(f"📊 Residual std: {np.std(residuals):.4f}")
