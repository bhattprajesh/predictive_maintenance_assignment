"""
Anomaly Detection System
Implements alert and error detection based on residual thresholds
"""

import numpy as np
import pandas as pd


class AnomalyDetector:
    """
    Detects anomalies in current consumption based on regression residuals
    """
    
    def __init__(self, MinC, MaxC, T):
        """
        Initialize detector with thresholds
        
        Args:
            MinC (float): Minimum current deviation for ALERT (kWh)
            MaxC (float): Maximum current deviation for ERROR (kWh)
            T (float): Minimum continuous time duration (seconds)
        """
        self.MinC = MinC
        self.MaxC = MaxC
        self.T = T
        
        print(f"🚨 Anomaly Detector Initialized:")
        print(f"   Alert Threshold (MinC): {MinC:.4f} kWh")
        print(f"   Error Threshold (MaxC): {MaxC:.4f} kWh")
        print(f"   Time Window (T): {T:.1f} seconds")
    
    def detect_anomalies(self, time, actual, predicted):
        """
        Detect alerts and errors based on thresholds
        
        Args:
            time (array-like): Timestamp array (seconds)
            actual (array-like): Actual current values (kWh)
            predicted (array-like): Predicted current values (kWh)
            
        Returns:
            tuple: (alerts, errors) - lists of anomaly dictionaries
        """
        time = np.array(time)
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Calculate residuals (positive = above expected)
        residuals = actual - predicted
        
        alerts = []
        errors = []
        
        # State tracking
        in_alert = False
        in_error = False
        alert_start_idx = None
        error_start_idx = None
        alert_max_dev = 0
        error_max_dev = 0
        
        for i in range(len(residuals)):
            deviation = residuals[i]
            
            # Priority 1: Check for ERROR (critical)
            if deviation >= self.MaxC:
                if not in_error:
                    # Start new error event
                    error_start_idx = i
                    in_error = True
                    error_max_dev = deviation
                else:
                    # Continue error event, update max deviation
                    error_max_dev = max(error_max_dev, deviation)
                
                # If we were in an alert, close it (error takes priority)
                if in_alert:
                    in_alert = False
                    alert_start_idx = None
                    
            else:
                # End error if it was active
                if in_error:
                    duration = time[i] - time[error_start_idx]
                    if duration >= self.T:
                        # Valid error (duration threshold met)
                        errors.append({
                            'start_time': time[error_start_idx],
                            'end_time': time[i-1],
                            'start_idx': error_start_idx,
                            'end_idx': i-1,
                            'duration': duration,
                            'max_deviation': error_max_dev,
                            'mean_deviation': np.mean(residuals[error_start_idx:i])
                        })
                    in_error = False
                    error_max_dev = 0
                
                # Priority 2: Check for ALERT (warning)
                if self.MinC <= deviation < self.MaxC:
                    if not in_alert:
                        # Start new alert event
                        alert_start_idx = i
                        in_alert = True
                        alert_max_dev = deviation
                    else:
                        # Continue alert event, update max deviation
                        alert_max_dev = max(alert_max_dev, deviation)
                else:
                    # End alert if it was active
                    if in_alert:
                        duration = time[i] - time[alert_start_idx]
                        if duration >= self.T:
                            # Valid alert (duration threshold met)
                            alerts.append({
                                'start_time': time[alert_start_idx],
                                'end_time': time[i-1],
                                'start_idx': alert_start_idx,
                                'end_idx': i-1,
                                'duration': duration,
                                'max_deviation': alert_max_dev,
                                'mean_deviation': np.mean(residuals[alert_start_idx:i])
                            })
                        in_alert = False
                        alert_max_dev = 0
        
        # Handle events that extend to end of data
        if in_error:
            duration = time[-1] - time[error_start_idx]
            if duration >= self.T:
                errors.append({
                    'start_time': time[error_start_idx],
                    'end_time': time[-1],
                    'start_idx': error_start_idx,
                    'end_idx': len(time)-1,
                    'duration': duration,
                    'max_deviation': error_max_dev,
                    'mean_deviation': np.mean(residuals[error_start_idx:])
                })
        
        if in_alert:
            duration = time[-1] - time[alert_start_idx]
            if duration >= self.T:
                alerts.append({
                    'start_time': time[alert_start_idx],
                    'end_time': time[-1],
                    'start_idx': alert_start_idx,
                    'end_idx': len(time)-1,
                    'duration': duration,
                    'max_deviation': alert_max_dev,
                    'mean_deviation': np.mean(residuals[alert_start_idx:])
                })
        
        return alerts, errors
    
    def detect_all_axes(self, test_data, models, time_column='Time'):
        """
        Detect anomalies across all axes
        
        Args:
            test_data (pd.DataFrame): Test data
            models (dict): Dictionary of trained regression models
            time_column (str): Name of time column
            
        Returns:
            pd.DataFrame: All anomaly events
        """
        all_events = []
        
        print("\n🔍 Detecting anomalies across all axes...\n")
        
        for axis_num, model in models.items():
            axis_col = f'Axis_{axis_num}'
            
            if axis_col not in test_data.columns:
                print(f"⚠️  Skipping {axis_col} - not in test data")
                continue
            
            # Get predictions and detect anomalies
            time = test_data[time_column].values
            actual = test_data[axis_col].values
            predicted = model.predict(time)
            
            alerts, errors = self.detect_anomalies(time, actual, predicted)
            
            # Add to all events
            for alert in alerts:
                all_events.append({
                    'axis': axis_num,
                    'type': 'ALERT',
                    'start_time': alert['start_time'],
                    'end_time': alert['end_time'],
                    'duration': alert['duration'],
                    'max_deviation': alert['max_deviation'],
                    'mean_deviation': alert['mean_deviation']
                })
            
            for error in errors:
                all_events.append({
                    'axis': axis_num,
                    'type': 'ERROR',
                    'start_time': error['start_time'],
                    'end_time': error['end_time'],
                    'duration': error['duration'],
                    'max_deviation': error['max_deviation'],
                    'mean_deviation': error['mean_deviation']
                })
            
            print(f"Axis {axis_num}: {len(alerts)} alerts, {len(errors)} errors")
        
        events_df = pd.DataFrame(all_events)
        
        if len(events_df) > 0:
            events_df = events_df.sort_values('start_time').reset_index(drop=True)
            
            print(f"\n✅ Total Anomalies Detected: {len(events_df)}")
            print(f"   Alerts: {len(events_df[events_df['type']=='ALERT'])}")
            print(f"   Errors: {len(events_df[events_df['type']=='ERROR'])}")
        else:
            print("\n⚠️  No anomalies detected with current thresholds")
        
        return events_df
    
    def get_summary_statistics(self, events_df):
        """
        Get summary statistics of detected anomalies
        
        Args:
            events_df (pd.DataFrame): DataFrame of anomaly events
            
        Returns:
            dict: Summary statistics
        """
        if len(events_df) == 0:
            return {"message": "No anomalies detected"}
        
        summary = {
            'total_events': len(events_df),
            'total_alerts': len(events_df[events_df['type'] == 'ALERT']),
            'total_errors': len(events_df[events_df['type'] == 'ERROR']),
            'avg_alert_duration': events_df[events_df['type'] == 'ALERT']['duration'].mean() if len(events_df[events_df['type'] == 'ALERT']) > 0 else 0,
            'avg_error_duration': events_df[events_df['type'] == 'ERROR']['duration'].mean() if len(events_df[events_df['type'] == 'ERROR']) > 0 else 0,
            'max_deviation_alert': events_df[events_df['type'] == 'ALERT']['max_deviation'].max() if len(events_df[events_df['type'] == 'ALERT']) > 0 else 0,
            'max_deviation_error': events_df[events_df['type'] == 'ERROR']['max_deviation'].max() if len(events_df[events_df['type'] == 'ERROR']) > 0 else 0,
            'events_by_axis': events_df.groupby('axis').size().to_dict()
        }
        
        return summary


def discover_thresholds_from_residuals(residuals, percentile_alert=95, percentile_error=99):
    """
    Discover appropriate thresholds from residual analysis
    
    Args:
        residuals (array-like): Residual values
        percentile_alert (float): Percentile for alert threshold (default 95)
        percentile_error (float): Percentile for error threshold (default 99)
        
    Returns:
        dict: Suggested thresholds with statistics
    """
    residuals = np.array(residuals)
    
    mean = np.mean(residuals)
    std = np.std(residuals)
    
    # Statistical thresholds
    two_sigma = mean + 2 * std
    three_sigma = mean + 3 * std
    
    # Percentile thresholds
    p95 = np.percentile(residuals, percentile_alert)
    p99 = np.percentile(residuals, percentile_error)
    
    # Count outliers
    outliers_2sigma = np.sum(residuals > two_sigma)
    outliers_3sigma = np.sum(residuals > three_sigma)
    
    suggestions = {
        'mean': mean,
        'std': std,
        'two_sigma_threshold': two_sigma,
        'three_sigma_threshold': three_sigma,
        'percentile_95': p95,
        'percentile_99': p99,
        'outliers_2sigma': outliers_2sigma,
        'outliers_3sigma': outliers_3sigma,
        'outliers_2sigma_pct': outliers_2sigma / len(residuals) * 100,
        'outliers_3sigma_pct': outliers_3sigma / len(residuals) * 100,
        'suggested_MinC': max(two_sigma, p95),  # Conservative: use higher value
        'suggested_MaxC': max(three_sigma, p99)  # Conservative: use higher value
    }
    
    print("\n📊 Threshold Discovery Analysis:")
    print(f"   Mean residual: {mean:.4f}")
    print(f"   Std residual: {std:.4f}")
    print(f"   2σ threshold: {two_sigma:.4f} ({outliers_2sigma} outliers, {suggestions['outliers_2sigma_pct']:.2f}%)")
    print(f"   3σ threshold: {three_sigma:.4f} ({outliers_3sigma} outliers, {suggestions['outliers_3sigma_pct']:.2f}%)")
    print(f"   95th percentile: {p95:.4f}")
    print(f"   99th percentile: {p99:.4f}")
    print(f"\n💡 Suggested Thresholds:")
    print(f"   MinC (Alert): {suggestions['suggested_MinC']:.4f} kWh")
    print(f"   MaxC (Error): {suggestions['suggested_MaxC']:.4f} kWh")
    
    return suggestions


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing AnomalyDetector...\n")
    
    # Generate sample data with anomalies
    np.random.seed(42)
    n = 500
    time = np.linspace(0, 500, n)
    
    # Normal pattern
    actual = 2.5 * time + 50 + np.random.normal(0, 5, n)
    predicted = 2.5 * time + 50
    
    # Inject anomalies
    actual[100:150] += 15  # Alert region
    actual[300:320] += 25  # Error region
    
    # Discover thresholds
    residuals = actual - predicted
    suggestions = discover_thresholds_from_residuals(residuals)
    
    # Create detector
    detector = AnomalyDetector(
        MinC=suggestions['suggested_MinC'],
        MaxC=suggestions['suggested_MaxC'],
        T=30
    )
    
    # Detect anomalies
    alerts, errors = detector.detect_anomalies(time, actual, predicted)
    
    print(f"\n✅ Detected: {len(alerts)} alerts, {len(errors)} errors")
    print("\n✅ Test completed successfully!")
