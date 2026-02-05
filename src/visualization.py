"""
Visualization utilities for predictive maintenance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


def plot_regression_with_alerts(time, actual, predicted, alerts, errors, 
                                axis_num, save_path=None):
    """
    Plot regression line with highlighted alert and error regions
    
    Args:
        time (array): Time values
        actual (array): Actual current values
        predicted (array): Predicted current values
        alerts (list): List of alert dictionaries
        errors (list): List of error dictionaries
        axis_num (int): Axis number
        save_path (str): Path to save figure (optional)
    """
    plt.figure(figsize=(16, 7))
    
    # Plot actual data and regression line
    plt.scatter(time, actual, alpha=0.4, s=20, label='Actual Current', color='steelblue')
    plt.plot(time, predicted, 'r-', label='Regression Line', linewidth=2.5)
    
    # Highlight alert regions (yellow)
    for alert in alerts:
        start_idx = alert['start_idx']
        end_idx = alert['end_idx']
        
        # Fill region
        plt.fill_between(time[start_idx:end_idx+1], 
                        actual[start_idx:end_idx+1], 
                        predicted[start_idx:end_idx+1],
                        color='yellow', alpha=0.4, label='Alert' if alerts.index(alert) == 0 else '')
        
        # Add vertical line and annotation
        plt.axvline(time[start_idx], color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
        plt.text(time[start_idx], max(actual) * 0.95, 
                f"⚠️ Alert\n{alert['duration']:.0f}s", 
                rotation=0, fontsize=9, ha='left', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Highlight error regions (red)
    for error in errors:
        start_idx = error['start_idx']
        end_idx = error['end_idx']
        
        # Fill region
        plt.fill_between(time[start_idx:end_idx+1], 
                        actual[start_idx:end_idx+1], 
                        predicted[start_idx:end_idx+1],
                        color='red', alpha=0.3, label='Error' if errors.index(error) == 0 else '')
        
        # Add vertical line and annotation
        plt.axvline(time[start_idx], color='darkred', linestyle='--', alpha=0.8, linewidth=2)
        plt.text(time[start_idx], max(actual) * 0.90, 
                f"🚨 ERROR\n{error['duration']:.0f}s", 
                rotation=0, fontsize=9, ha='left', weight='bold',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.7, edgecolor='darkred'))
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Current (kWh)', fontsize=12)
    plt.title(f'Axis {axis_num}: Regression with Alert/Error Detection', fontsize=14, weight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot: {save_path}")
    
    plt.show()


def plot_residual_analysis(time, residuals, axis_num, save_path=None):
    """
    Plot residual analysis (scatter, histogram, boxplot)
    
    Args:
        time (array): Time values
        residuals (array): Residual values
        axis_num (int): Axis number
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Scatter plot
    axes[0].scatter(time, residuals, alpha=0.5, s=15, color='steelblue')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Line')
    axes[0].axhline(y=np.mean(residuals), color='green', linestyle='--', linewidth=1.5, label='Mean')
    axes[0].axhline(y=np.mean(residuals) + 2*np.std(residuals), color='orange', linestyle=':', linewidth=1.5, label='±2σ')
    axes[0].axhline(y=np.mean(residuals) - 2*np.std(residuals), color='orange', linestyle=':', linewidth=1.5)
    axes[0].set_xlabel('Time (seconds)', fontsize=11)
    axes[0].set_ylabel('Residual (kWh)', fontsize=11)
    axes[0].set_title(f'Axis {axis_num}: Residuals Over Time', fontsize=12, weight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[1].axvline(x=np.mean(residuals), color='green', linestyle='--', linewidth=1.5, label='Mean')
    axes[1].set_xlabel('Residual (kWh)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=12, weight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Boxplot
    bp = axes[2].boxplot(residuals, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    axes[2].set_ylabel('Residual (kWh)', fontsize=11)
    axes[2].set_title('Residual Boxplot', fontsize=12, weight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"Mean: {np.mean(residuals):.4f}\nStd: {np.std(residuals):.4f}\n95%: {np.percentile(residuals, 95):.4f}\n99%: {np.percentile(residuals, 99):.4f}"
    axes[2].text(1.3, np.median(residuals), stats_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot: {save_path}")
    
    plt.show()


def plot_all_axes_comparison(models, test_data, time_column='Time', save_path=None):
    """
    Plot all 8 axes in a grid for comparison
    
    Args:
        models (dict): Dictionary of regression models
        test_data (pd.DataFrame): Test data
        time_column (str): Name of time column
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    for i, (axis_num, model) in enumerate(models.items()):
        axis_col = f'Axis_{axis_num}'
        
        if axis_col not in test_data.columns:
            continue
        
        time = test_data[time_column].values
        actual = test_data[axis_col].values
        predicted = model.predict(time)
        
        axes[i].scatter(time, actual, alpha=0.4, s=10, label='Actual', color='steelblue')
        axes[i].plot(time, predicted, 'r-', label='Predicted', linewidth=2)
        axes[i].set_xlabel('Time (s)', fontsize=10)
        axes[i].set_ylabel('Current (kWh)', fontsize=10)
        axes[i].set_title(f'Axis {axis_num} (R²={model.r2_score:.3f})', fontsize=11, weight='bold')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot: {save_path}")
    
    plt.show()


def plot_anomaly_summary(events_df, save_path=None):
    """
    Plot summary of anomalies across all axes
    
    Args:
        events_df (pd.DataFrame): DataFrame of anomaly events
        save_path (str): Path to save figure
    """
    if len(events_df) == 0:
        print("⚠️  No anomalies to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Anomalies by axis
    anomaly_counts = events_df.groupby(['axis', 'type']).size().unstack(fill_value=0)
    anomaly_counts.plot(kind='bar', ax=axes[0, 0], color=['orange', 'red'], alpha=0.7)
    axes[0, 0].set_xlabel('Axis', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Anomaly Count by Axis', fontsize=12, weight='bold')
    axes[0, 0].legend(['Alert', 'Error'])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Duration distribution
    alerts_duration = events_df[events_df['type'] == 'ALERT']['duration']
    errors_duration = events_df[events_df['type'] == 'ERROR']['duration']
    
    if len(alerts_duration) > 0:
        axes[0, 1].hist(alerts_duration, bins=15, alpha=0.6, label='Alerts', color='orange', edgecolor='black')
    if len(errors_duration) > 0:
        axes[0, 1].hist(errors_duration, bins=15, alpha=0.6, label='Errors', color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Duration (seconds)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Anomaly Duration Distribution', fontsize=12, weight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Timeline of anomalies
    for idx, row in events_df.iterrows():
        color = 'orange' if row['type'] == 'ALERT' else 'red'
        axes[1, 0].barh(row['axis'], row['duration'], left=row['start_time'], 
                       color=color, alpha=0.6, edgecolor='black')
    axes[1, 0].set_xlabel('Time (seconds)', fontsize=11)
    axes[1, 0].set_ylabel('Axis', fontsize=11)
    axes[1, 0].set_title('Anomaly Timeline', fontsize=12, weight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 4. Deviation distribution
    axes[1, 1].boxplot([events_df[events_df['type'] == 'ALERT']['max_deviation'],
                       events_df[events_df['type'] == 'ERROR']['max_deviation']],
                      labels=['Alert', 'Error'],
                      patch_artist=True)
    axes[1, 1].set_ylabel('Max Deviation (kWh)', fontsize=11)
    axes[1, 1].set_title('Deviation Distribution by Type', fontsize=12, weight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot: {save_path}")
    
    plt.show()


def create_results_summary_report(models, events_df, thresholds, output_path='results/summary_report.txt'):
    """
    Create a text summary report of all results
    
    Args:
        models (dict): Regression models
        events_df (pd.DataFrame): Anomaly events
        thresholds (dict): Detection thresholds
        output_path (str): Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PREDICTIVE MAINTENANCE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Model Performance
        f.write("REGRESSION MODEL PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        for axis_num, model in models.items():
            f.write(f"Axis {axis_num}:\n")
            f.write(f"  Slope: {model.slope:.6f}\n")
            f.write(f"  Intercept: {model.intercept:.6f}\n")
            f.write(f"  R² Score: {model.r2_score:.4f}\n")
            f.write(f"  RMSE: {model.rmse:.4f}\n\n")
        
        avg_r2 = np.mean([m.r2_score for m in models.values()])
        f.write(f"Average R² Score: {avg_r2:.4f}\n\n")
        
        # Thresholds
        f.write("DETECTION THRESHOLDS\n")
        f.write("-" * 80 + "\n")
        f.write(f"MinC (Alert): {thresholds['MinC']:.4f} kWh\n")
        f.write(f"MaxC (Error): {thresholds['MaxC']:.4f} kWh\n")
        f.write(f"T (Time Window): {thresholds['T']:.1f} seconds\n\n")
        
        # Anomaly Detection Results
        f.write("ANOMALY DETECTION RESULTS\n")
        f.write("-" * 80 + "\n")
        if len(events_df) > 0:
            total_alerts = len(events_df[events_df['type'] == 'ALERT'])
            total_errors = len(events_df[events_df['type'] == 'ERROR'])
            
            f.write(f"Total Anomalies: {len(events_df)}\n")
            f.write(f"  Alerts: {total_alerts}\n")
            f.write(f"  Errors: {total_errors}\n\n")
            
            f.write("Anomalies by Axis:\n")
            for axis in sorted(events_df['axis'].unique()):
                axis_events = events_df[events_df['axis'] == axis]
                axis_alerts = len(axis_events[axis_events['type'] == 'ALERT'])
                axis_errors = len(axis_events[axis_events['type'] == 'ERROR'])
                f.write(f"  Axis {axis}: {axis_alerts} alerts, {axis_errors} errors\n")
            
            f.write(f"\nAverage Alert Duration: {events_df[events_df['type']=='ALERT']['duration'].mean():.2f} seconds\n")
            f.write(f"Average Error Duration: {events_df[events_df['type']=='ERROR']['duration'].mean():.2f} seconds\n")
        else:
            f.write("No anomalies detected\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"📄 Summary report saved to {output_path}")


if __name__ == "__main__":
    print("✅ Visualization module loaded successfully")
