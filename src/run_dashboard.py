"""
Standalone Predictive Maintenance Dashboard Launcher
Run this to generate the 2-week forecast dashboard
"""

import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
from regression import MultiAxisRegressionSystem
from anomaly_detection import AnomalyDetector
from predictive_dashboard import PredictiveMaintenanceDashboard
import os

print("=" * 80)
print("🎯 PREDICTIVE MAINTENANCE DASHBOARD LAUNCHER")
print("=" * 80)
print()

# Configuration
TRAIN_DATA_PATH = '../data/processed/robot_data_clean.csv'
TEST_DATA_PATH = '../data/processed/synthetic_test_data.csv'
MODELS_DIR = '../models'
RESULTS_DIR = '../results'

# Thresholds (adjust based on your analysis)
MinC = 0.5  # Alert threshold (A)
MaxC = 1.0  # Error threshold (A)
T = 60      # Time window (seconds)
FORECAST_DAYS = 14  # 2 weeks

print("📂 Loading data...")

# Load training data
if not os.path.exists(TRAIN_DATA_PATH):
    print(f"❌ Training data not found at {TRAIN_DATA_PATH}")
    print("   Please run the notebook first to preprocess data")
    sys.exit(1)

training_data = pd.read_csv(TRAIN_DATA_PATH)
print(f"✅ Loaded training data: {len(training_data)} rows")

# Load test data
if not os.path.exists(TEST_DATA_PATH):
    print(f"⚠️  Test data not found. Generating synthetic data...")
    from data_generator import SyntheticDataGenerator
    
    generator = SyntheticDataGenerator(training_data)
    test_data = generator.generate_test_data(n_samples=1000, anomaly_rate=0.10)
    test_data.to_csv(TEST_DATA_PATH, index=False)
    print(f"✅ Generated test data: {len(test_data)} rows")
else:
    test_data = pd.read_csv(TEST_DATA_PATH)
    print(f"✅ Loaded test data: {len(test_data)} rows")

print("\n🤖 Loading regression models...")

# Load or train models
regression_system = MultiAxisRegressionSystem()

if os.path.exists(MODELS_DIR) and os.listdir(MODELS_DIR):
    print("   Loading pre-trained models...")
    regression_system.load_all_models(MODELS_DIR)
    print(f"✅ Loaded {len(regression_system.models)} models")
else:
    print("   Training new models...")
    regression_system.train_all_axes(training_data, time_column='Time')
    
    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    regression_system.save_all_models(MODELS_DIR)
    print(f"✅ Trained and saved {len(regression_system.models)} models")

print(f"\n⚙️  Initializing anomaly detector...")
print(f"   MinC (Alert): {MinC} A")
print(f"   MaxC (Error): {MaxC} A")
print(f"   Time window: {T} seconds")

detector = AnomalyDetector(MinC=MinC, MaxC=MaxC, T=T)

print(f"\n🎯 Creating predictive dashboard...")
print(f"   Forecast horizon: {FORECAST_DAYS} days")

dashboard = PredictiveMaintenanceDashboard(
    models=regression_system.models,
    detector=detector,
    forecast_days=FORECAST_DAYS
)

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n📊 Generating dashboard visualizations...")
print("   This may take a minute...")

# Generate main dashboard
predictions = dashboard.create_dashboard(
    test_data,
    save_path=os.path.join(RESULTS_DIR, 'predictive_dashboard.png')
)

print(f"\n{'=' * 80}")
print("📋 PREDICTION SUMMARY")
print(f"{'=' * 80}")

if predictions:
    print(f"\n⚠️  Found {len(predictions)} predicted failure(s):\n")
    
    for i, pred in enumerate(predictions, 1):
        severity_emoji = '🔴' if pred['severity'] == 'CRITICAL' else '🟡' if pred['severity'] == 'WARNING' else '🟠'
        print(f"{i}. {severity_emoji} Axis {pred['axis']}")
        print(f"   Severity: {pred['severity']}")
        print(f"   Time to failure: {pred['time_to_failure_days']:.1f} days")
        print(f"   Current deviation: {pred['current_deviation']:.3f} A")
        print(f"   Predicted peak: {pred['predicted_peak_deviation']:.3f} A")
        print()
    
    # Generate alert pop-up
    print("🚨 Generating alert pop-up...")
    dashboard.generate_alert_popup(
        predictions,
        save_path=os.path.join(RESULTS_DIR, 'alert_popup.png')
    )
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.join(RESULTS_DIR, 'failure_predictions.csv'), index=False)
    print(f"💾 Predictions saved to {RESULTS_DIR}/failure_predictions.csv")
    
else:
    print("\n✅ No failures predicted within {FORECAST_DAYS}-day forecast horizon")
    print("   All systems operating normally")

print(f"\n{'=' * 80}")
print("✅ DASHBOARD GENERATION COMPLETE")
print(f"{'=' * 80}")
print("\nGenerated files:")
print(f"  - {RESULTS_DIR}/predictive_dashboard.png")
if predictions:
    print(f"  - {RESULTS_DIR}/alert_popup.png")
    print(f"  - {RESULTS_DIR}/failure_predictions.csv")

print("\n💡 Next steps:")
print("  1. Review the dashboard visualization")
print("  2. Check alert pop-up for urgent actions")
print("  3. Schedule maintenance based on predictions")
print("  4. Update thresholds if needed based on results")

print("\n" + "=" * 80)
