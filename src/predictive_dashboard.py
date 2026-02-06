"""
Real-Time Predictive Maintenance Dashboard
Features:
- Live monitoring of all 8 axes
- 2-week ahead failure prediction
- Pop-up alerts for predicted failures
- Time-to-failure estimation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class PredictiveMaintenanceDashboard:
    """
    Interactive dashboard for predictive maintenance with 2-week forecasting
    """
    
    def __init__(self, models, detector, forecast_days=14):
        """
        Initialize dashboard
        
        Args:
            models (dict): Trained regression models
            detector (AnomalyDetector): Anomaly detector with thresholds
            forecast_days (int): Days to forecast ahead (default 14)
        """
        self.models = models
        self.detector = detector
        self.forecast_days = forecast_days
        self.forecast_seconds = forecast_days * 24 * 3600  # Convert to seconds
        
        # Alert tracking
        self.active_alerts = []
        self.predicted_failures = []
        
        print(f"🎯 Predictive Dashboard Initialized")
        print(f"   Forecast horizon: {forecast_days} days ({self.forecast_seconds/3600:.0f} hours)")
        print(f"   Alert threshold: {detector.MinC:.4f} A")
        print(f"   Error threshold: {detector.MaxC:.4f} A")
    
    def forecast_trend(self, time_history, current_history, forecast_steps=1000):
        """
        Forecast future current consumption using trend analysis
        
        Args:
            time_history (array): Historical time values
            current_history (array): Historical current values
            forecast_steps (int): Number of future time steps to predict
            
        Returns:
            tuple: (future_times, forecasted_currents, confidence_bounds)
        """
        # Fit trend on recent data (last 1000 points for better accuracy)
        recent_points = min(1000, len(time_history))
        recent_time = time_history[-recent_points:]
        recent_current = current_history[-recent_points:]
        
        # Calculate trend (slope)
        coeffs = np.polyfit(recent_time, recent_current, deg=1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Calculate residual standard deviation for confidence bounds
        fitted = slope * recent_time + intercept
        residuals = recent_current - fitted
        std_residual = np.std(residuals)
        
        # Generate future time points
        last_time = time_history[-1]
        future_times = np.linspace(last_time, last_time + self.forecast_seconds, forecast_steps)
        
        # Forecast using linear trend
        forecasted_currents = slope * future_times + intercept
        
        # Confidence bounds (95% confidence interval)
        upper_bound = forecasted_currents + 2 * std_residual
        lower_bound = forecasted_currents - 2 * std_residual
        
        return future_times, forecasted_currents, (lower_bound, upper_bound)
    
    def predict_failure_time(self, axis_num, time_history, current_history):
        """
        Predict when this axis will fail (exceed MaxC threshold)
        
        Args:
            axis_num (int): Axis number
            time_history (array): Historical time values
            current_history (array): Historical current values
            
        Returns:
            dict: Failure prediction info or None
        """
        # Get model predictions
        model = self.models[axis_num]
        predicted_baseline = model.predict(time_history)
        
        # Calculate current deviation trend
        deviations = current_history - predicted_baseline
        
        # Forecast future deviations
        future_times, future_currents, (lower, upper) = self.forecast_trend(
            time_history, current_history
        )
        
        future_baseline = model.predict(future_times)
        future_deviations = future_currents - future_baseline
        
        # Find when deviation will exceed MaxC
        critical_indices = np.where(future_deviations >= self.detector.MaxC)[0]
        
        if len(critical_indices) > 0:
            # Time until failure
            failure_time = future_times[critical_indices[0]]
            current_time = time_history[-1]
            time_to_failure = failure_time - current_time
            
            # Convert to days
            days_to_failure = time_to_failure / (24 * 3600)
            
            # Only alert if within forecast horizon
            if days_to_failure <= self.forecast_days:
                return {
                    'axis': axis_num,
                    'time_to_failure_seconds': time_to_failure,
                    'time_to_failure_days': days_to_failure,
                    'predicted_failure_time': failure_time,
                    'current_deviation': deviations[-1],
                    'predicted_peak_deviation': future_deviations[critical_indices[0]],
                    'severity': 'CRITICAL' if days_to_failure < 7 else 'WARNING'
                }
        
        # Check for alert level (MinC)
        alert_indices = np.where(future_deviations >= self.detector.MinC)[0]
        if len(alert_indices) > 0:
            alert_time = future_times[alert_indices[0]]
            current_time = time_history[-1]
            time_to_alert = alert_time - current_time
            days_to_alert = time_to_alert / (24 * 3600)
            
            if days_to_alert <= self.forecast_days:
                return {
                    'axis': axis_num,
                    'time_to_failure_seconds': time_to_alert,
                    'time_to_failure_days': days_to_alert,
                    'predicted_failure_time': alert_time,
                    'current_deviation': deviations[-1],
                    'predicted_peak_deviation': future_deviations[alert_indices[0]],
                    'severity': 'ALERT'
                }
        
        return None
    
    def create_dashboard(self, test_data, save_path='../results/predictive_dashboard.png'):
        """
        Create comprehensive predictive maintenance dashboard
        
        Args:
            test_data (pd.DataFrame): Test data to visualize
            save_path (str): Path to save dashboard image
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle('🎯 Predictive Maintenance Dashboard - RMBR4-2 Robot\n2-Week Failure Forecast System', 
                     fontsize=18, weight='bold', y=0.98)
        
        # Current timestamp
        current_time = datetime.now()
        forecast_end = current_time + timedelta(days=self.forecast_days)
        
        fig.text(0.5, 0.94, f'Current Time: {current_time.strftime("%Y-%m-%d %H:%M:%S")} | '
                            f'Forecast Until: {forecast_end.strftime("%Y-%m-%d")}',
                 ha='center', fontsize=12, style='italic')
        
        # Analyze each axis and predict failures
        predictions = []
        
        for i, (axis_num, model) in enumerate(self.models.items()):
            row = i // 3
            col = i % 3
            
            # Create subplot for this axis
            if row < 2:  # First 2 rows for individual axis plots
                ax = fig.add_subplot(gs[row, col])
            else:  # Row 2 onwards for additional info
                continue
            
            axis_col = f'Axis_{axis_num}'
            time = test_data['Time'].values
            actual = test_data[axis_col].values
            predicted_baseline = model.predict(time)
            
            # Calculate deviation
            deviation = actual - predicted_baseline
            
            # Forecast future
            future_times, future_currents, (lower, upper) = self.forecast_trend(time, actual)
            future_baseline = model.predict(future_times)
            future_deviation = future_currents - future_baseline
            
            # Predict failure
            failure_pred = self.predict_failure_time(axis_num, time, actual)
            if failure_pred:
                predictions.append(failure_pred)
            
            # Plot historical data
            ax.plot(time, actual, 'b-', alpha=0.6, linewidth=1, label='Actual')
            ax.plot(time, predicted_baseline, 'g--', linewidth=2, label='Expected (Baseline)')
            
            # Plot forecast
            ax.plot(future_times, future_currents, 'r--', linewidth=2, label='Forecast')
            ax.fill_between(future_times, lower, upper, color='red', alpha=0.1, label='95% Confidence')
            
            # Threshold lines
            last_time_extended = np.concatenate([time, future_times])
            ax.axhline(y=predicted_baseline[-1] + self.detector.MinC, 
                      color='orange', linestyle=':', linewidth=2, label='Alert Threshold')
            ax.axhline(y=predicted_baseline[-1] + self.detector.MaxC, 
                      color='red', linestyle=':', linewidth=2, label='Failure Threshold')
            
            # Mark current time
            ax.axvline(x=time[-1], color='black', linestyle='-', linewidth=2, alpha=0.5)
            
            # Highlight failure prediction
            if failure_pred:
                failure_time = failure_pred['predicted_failure_time']
                ax.axvline(x=failure_time, color='red', linestyle='--', linewidth=3, alpha=0.7)
                
                # Add text annotation
                days = failure_pred['time_to_failure_days']
                severity = failure_pred['severity']
                color = 'red' if severity == 'CRITICAL' else 'orange'
                
                ax.text(failure_time, ax.get_ylim()[1] * 0.9, 
                       f'⚠️ {severity}\n{days:.1f} days',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                       fontsize=9, weight='bold', ha='center')
            
            ax.set_xlabel('Time (seconds)', fontsize=9)
            ax.set_ylabel('Current (A)', fontsize=9)
            ax.set_title(f'Axis {axis_num}', fontsize=11, weight='bold')
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Summary panel (bottom section)
        ax_summary = fig.add_subplot(gs[2:4, :])
        ax_summary.axis('off')
        
        # Create alerts table
        if predictions:
            predictions_sorted = sorted(predictions, key=lambda x: x['time_to_failure_days'])
            
            # Table data
            table_data = []
            for pred in predictions_sorted:
                severity = pred['severity']
                emoji = '🔴' if severity == 'CRITICAL' else '🟡' if severity == 'WARNING' else '🟠'
                
                table_data.append([
                    f"{emoji} Axis {pred['axis']}",
                    severity,
                    f"{pred['time_to_failure_days']:.1f} days",
                    f"{pred['current_deviation']:.3f} A",
                    f"{pred['predicted_peak_deviation']:.3f} A"
                ])
            
            # Create table
            table = ax_summary.table(
                cellText=table_data,
                colLabels=['Axis', 'Severity', 'Time to Failure', 'Current Deviation', 'Predicted Peak'],
                cellLoc='center',
                loc='center',
                bbox=[0.1, 0.3, 0.8, 0.6]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Color code by severity
            for i, pred in enumerate(predictions_sorted):
                severity = pred['severity']
                if severity == 'CRITICAL':
                    color = '#ffcccc'
                elif severity == 'WARNING':
                    color = '#fff4cc'
                else:
                    color = '#ffe6cc'
                
                for j in range(5):
                    table[(i+1, j)].set_facecolor(color)
            
            # Header styling
            for j in range(5):
                table[(0, j)].set_facecolor('#4CAF50')
                table[(0, j)].set_text_props(weight='bold', color='white')
            
            # Summary statistics
            critical_count = sum(1 for p in predictions if p['severity'] == 'CRITICAL')
            warning_count = sum(1 for p in predictions if p['severity'] == 'WARNING')
            alert_count = sum(1 for p in predictions if p['severity'] == 'ALERT')
            
            summary_text = (
                f"📊 PREDICTIVE MAINTENANCE SUMMARY\n\n"
                f"🔴 Critical Failures (< 7 days): {critical_count}\n"
                f"🟡 Warnings (7-14 days): {warning_count}\n"
                f"🟠 Alerts (elevated risk): {alert_count}\n\n"
                f"⏰ Most Urgent: Axis {predictions_sorted[0]['axis']} "
                f"({predictions_sorted[0]['time_to_failure_days']:.1f} days)"
            )
        else:
            summary_text = (
                f"✅ ALL SYSTEMS NORMAL\n\n"
                f"No failures predicted within {self.forecast_days}-day forecast horizon.\n"
                f"All axes operating within acceptable parameters.\n\n"
                f"Next scheduled check: {(current_time + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')}"
            )
        
        ax_summary.text(0.5, 0.05, summary_text,
                       ha='center', va='bottom', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Recommendation panel
        ax_recommend = fig.add_subplot(gs[4, :])
        ax_recommend.axis('off')
        
        if predictions and predictions_sorted[0]['time_to_failure_days'] < 7:
            recommendation = (
                "🚨 IMMEDIATE ACTION REQUIRED\n\n"
                f"Schedule maintenance for Axis {predictions_sorted[0]['axis']} within "
                f"{int(predictions_sorted[0]['time_to_failure_days'])} days.\n"
                "Recommended actions: Inspect bearings, check lubrication, test motor windings.\n"
                "Contact: Maintenance Team | Priority: HIGH"
            )
            bg_color = '#ffcccc'
        elif predictions:
            recommendation = (
                "⚠️ MAINTENANCE RECOMMENDED\n\n"
                f"Plan maintenance for {len(predictions)} axis/axes within forecast period.\n"
                "Schedule during next planned downtime window.\n"
                "Contact: Maintenance Team | Priority: MEDIUM"
            )
            bg_color = '#fff4cc'
        else:
            recommendation = (
                "✅ NO ACTION REQUIRED\n\n"
                "Continue normal operations. Next review in 24 hours.\n"
                "All predictive indicators within normal ranges."
            )
            bg_color = '#ccffcc'
        
        ax_recommend.text(0.5, 0.5, recommendation,
                         ha='center', va='center', fontsize=11,
                         bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5, pad=1))
        
        # Save dashboard
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Dashboard saved to {save_path}")
        
        plt.show()
        
        return predictions
    
    def generate_alert_popup(self, predictions, save_path='../results/alert_popup.png'):
        """
        Generate pop-up style alert visualization
        
        Args:
            predictions (list): List of failure predictions
            save_path (str): Path to save alert image
        """
        if not predictions:
            print("✅ No alerts to display - all systems normal")
            return
        
        # Sort by urgency
        predictions_sorted = sorted(predictions, key=lambda x: x['time_to_failure_days'])
        
        # Create pop-up figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Background
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                        facecolor='#fff5f5', edgecolor='red', linewidth=3)
        ax.add_patch(rect)
        
        # Title
        most_urgent = predictions_sorted[0]
        title_text = "🚨 PREDICTIVE MAINTENANCE ALERT 🚨"
        ax.text(0.5, 0.92, title_text,
               ha='center', va='top', fontsize=24, weight='bold',
               color='darkred', transform=ax.transAxes)
        
        # Main alert message
        if most_urgent['time_to_failure_days'] < 3:
            urgency = "CRITICAL - IMMEDIATE ACTION REQUIRED"
            color = 'darkred'
        elif most_urgent['time_to_failure_days'] < 7:
            urgency = "HIGH PRIORITY - ACTION REQUIRED WITHIN WEEK"
            color = 'red'
        else:
            urgency = "MEDIUM PRIORITY - PLAN MAINTENANCE"
            color = 'orange'
        
        ax.text(0.5, 0.82, urgency,
               ha='center', va='top', fontsize=16, weight='bold',
               color=color, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Details
        y_pos = 0.70
        for i, pred in enumerate(predictions_sorted[:5]):  # Show top 5
            severity_emoji = '🔴' if pred['severity'] == 'CRITICAL' else '🟡' if pred['severity'] == 'WARNING' else '🟠'
            
            alert_text = (
                f"{severity_emoji} Axis {pred['axis']}: Predicted failure in {pred['time_to_failure_days']:.1f} days\n"
                f"   Current deviation: {pred['current_deviation']:.3f} A | "
                f"Predicted peak: {pred['predicted_peak_deviation']:.3f} A"
            )
            
            ax.text(0.1, y_pos, alert_text,
                   ha='left', va='top', fontsize=12,
                   transform=ax.transAxes, family='monospace')
            
            y_pos -= 0.12
        
        # Timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ax.text(0.5, 0.15, f"Alert Generated: {timestamp}",
               ha='center', va='top', fontsize=10, style='italic',
               transform=ax.transAxes)
        
        # Action required
        action_text = (
            "📋 REQUIRED ACTIONS:\n"
            f"1. Schedule inspection of Axis {most_urgent['axis']}\n"
            "2. Prepare replacement parts inventory\n"
            "3. Notify maintenance team and plan downtime\n"
            "4. Review historical data for root cause analysis"
        )
        
        ax.text(0.5, 0.08, action_text,
               ha='center', va='top', fontsize=11,
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🚨 Alert popup saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("✅ Predictive Dashboard module loaded")
    print("   Use: PredictiveMaintenanceDashboard(models, detector)")
    print("   Then: dashboard.create_dashboard(test_data)")
