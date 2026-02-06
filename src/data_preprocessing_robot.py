"""
Data preprocessing for RMBR4-2 robot data
Handles the specific format with timestamps and axis numbering
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_and_preprocess_robot_data(filepath):
    """
    Load and preprocess the RMBR4-2 robot data
    
    Handles:
    - Column names with "#" (Axis #1, Axis #2, etc.)
    - ISO timestamp format
    - Trait column removal
    - Time conversion to seconds
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Preprocessed data with standardized column names
    """
    print(f"📂 Loading data from {filepath}...")
    
    # Load data
    df = pd.read_csv(filepath)
    
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"📊 Original columns: {list(df.columns)}")
    
    # Remove 'Trait' column if exists
    if 'Trait' in df.columns:
        df = df.drop('Trait', axis=1)
        print("✅ Removed 'Trait' column")
    
    # Convert timestamp to seconds from start
    if 'Time' in df.columns:
        # Parse ISO timestamps
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Convert to seconds from start
        start_time = df['Time'].iloc[0]
        df['Time'] = (df['Time'] - start_time).dt.total_seconds()
        
        print(f"✅ Converted timestamps to seconds (0 to {df['Time'].max():.2f}s)")
    
    # Rename axis columns to standard format (Axis #1 -> Axis_1)
    column_mapping = {}
    for col in df.columns:
        if col.startswith('Axis #'):
            # Extract number and create new name
            axis_num = col.replace('Axis #', '')
            new_name = f'Axis_{axis_num}'
            column_mapping[col] = new_name
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"✅ Renamed {len(column_mapping)} axis columns")
        print(f"   Example: 'Axis #1' -> 'Axis_1'")
    
    # Keep only first 8 axes and Time column
    axes_to_keep = [f'Axis_{i}' for i in range(1, 9)]
    columns_to_keep = ['Time'] + axes_to_keep
    
    # Filter to only columns that exist
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns]
    
    print(f"✅ Kept columns: {available_columns}")
    
    # Convert axis columns to numeric, handling any non-numeric values
    for col in axes_to_keep:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with NaN values
    initial_rows = len(df)
    df = df.dropna()
    removed_rows = initial_rows - len(df)
    
    if removed_rows > 0:
        print(f"⚠️  Removed {removed_rows} rows with missing values")
    
    print(f"\n📊 Final dataset shape: {df.shape}")
    print(f"📊 Time range: {df['Time'].min():.2f}s to {df['Time'].max():.2f}s")
    print(f"📊 Data summary:")
    print(df.describe())
    
    return df


def save_preprocessed_data(df, output_path):
    """
    Save preprocessed data to CSV
    
    Args:
        df (pd.DataFrame): Preprocessed data
        output_path (str): Output file path
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\n💾 Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing data preprocessing...\n")
    
    # Load and preprocess
    df = load_and_preprocess_robot_data('../data/raw/RMBR4-2_export_test.csv')
    
    # Save
    save_preprocessed_data(df, '../data/processed/robot_data_clean.csv')
    
    print("\n✅ Preprocessing complete!")
