#!/usr/bin/env python3
"""
MedChain AI Performance Dashboard
Interactive dashboard for visualizing model performance metrics
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import glob

# Add the monitoring directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from model_monitor import ModelMonitor
    print("✅ Successfully imported model monitor")
except ImportError as e:
    print(f"❌ Failed to import model monitor: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_dashboard")

class PerformanceDashboard:
    """
    Interactive dashboard for visualizing model performance metrics
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the dashboard
        
        Args:
            data_dir: Directory containing monitoring data
        """
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        else:
            self.data_dir = data_dir
            
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            logger.warning(f"Data directory not found, created: {self.data_dir}")
        
        # Metrics directory
        self.metrics_dir = os.path.join(self.data_dir, "metrics")
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Alerts directory
        self.alerts_dir = os.path.join(self.data_dir, "alerts")
        if not os.path.exists(self.alerts_dir):
            os.makedirs(self.alerts_dir, exist_ok=True)
        
        # Dashboard output directory
        self.output_dir = os.path.join(self.data_dir, "dashboards")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Load baseline metrics if available
        self.baseline_metrics = self._load_baseline_metrics()
        
        logger.info("Performance dashboard initialized")
    
    def _load_baseline_metrics(self) -> Optional[Dict]:
        """Load baseline metrics from file"""
        
        baseline_path = os.path.join(self.metrics_dir, "baseline.json")
        
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path, "r") as f:
                    baseline = json.load(f)
                logger.info(f"Loaded baseline metrics from {baseline_path}")
                return baseline
            except Exception as e:
                logger.error(f"Error loading baseline metrics: {e}")
        
        logger.warning("No baseline metrics found")
        return None
    
    def _load_metrics_data(self, days: int = 7) -> pd.DataFrame:
        """Load metrics data from files"""
        
        # Get all metrics files
        metrics_files = glob.glob(os.path.join(self.metrics_dir, "metrics_*.json"))
        
        if not metrics_files:
            logger.warning("No metrics files found")
            return pd.DataFrame()
        
        # Filter by date if needed
        if days > 0:
            cutoff_date = datetime.now() - timedelta(days=days)
            metrics_files = [
                f for f in metrics_files 
                if datetime.fromtimestamp(os.path.getmtime(f)) >= cutoff_date
            ]
        
        # Load data from files
        metrics_data = []
        
        for file_path in metrics_files:
            try:
                with open(file_path, "r") as f:
                    metrics = json.load(f)
                
                # Add file timestamp if not in metrics
                if "timestamp" not in metrics:
                    file_timestamp = os.path.basename(file_path).replace("metrics_", "").replace(".json", "")
                    try:
                        metrics["timestamp"] = datetime.strptime(file_timestamp, "%Y%m%d_%H%M%S").isoformat()
                    except:
                        metrics["timestamp"] = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                
                metrics_data.append(metrics)
            except Exception as e:
                logger.error(f"Error loading metrics from {file_path}: {e}")
        
        # Convert to DataFrame
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
            
            return df
        else:
            return pd.DataFrame()
    
    def _load_alerts_data(self, days: int = 7) -> pd.DataFrame:
        """Load alerts data from files"""
        
        # Get all alert files
        alert_files = glob.glob(os.path.join(self.alerts_dir, "alert_*.json"))
        
        if not alert_files:
            logger.warning("No alert files found")
            return pd.DataFrame()
        
        # Filter by date if needed
        if days > 0:
            cutoff_date = datetime.now() - timedelta(days=days)
            alert_files = [
                f for f in alert_files 
                if datetime.fromtimestamp(os.path.getmtime(f)) >= cutoff_date
            ]
        
        # Load data from files
        alerts_data = []
        
        for file_path in alert_files:
            try:
                with open(file_path, "r") as f:
                    alert = json.load(f)
                alerts_data.append(alert)
            except Exception as e:
                logger.error(f"Error loading alert from {file_path}: {e}")
        
        # Convert to DataFrame
        if alerts_data:
            df = pd.DataFrame(alerts_data)
            
            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
            
            return df
        else:
            return pd.DataFrame()
    
    def generate_performance_dashboard(self, days: int = 7, output_file: str = None) -> str:
        """
        Generate performance dashboard
        
        Args:
            days: Number of days of data to include
            output_file: Output file path
        
        Returns:
            Path to the generated dashboard file
        """
        
        logger.info(f"Generating performance dashboard for the last {days} days...")
        
        # Load metrics data
        metrics_df = self._load_metrics_data(days)
        
        if metrics_df.empty:
            logger.warning("No metrics data available")
            return None
        
        # Load alerts data
        alerts_df = self._load_alerts_data(days)
        
        # Set up the figure
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle('MedChain AI Performance Dashboard', fontsize=16)
        
        # Create subplots
        gs = fig.add_gridspec(3, 2)
        
        # Accuracy plot
        ax1 = fig.add_subplot(gs[0, 0])
        if "current_accuracy" in metrics_df.columns:
            metrics_df.plot(x="timestamp", y="current_accuracy", ax=ax1, color='blue', marker='o', markersize=4)
            if self.baseline_metrics and "mean_accuracy" in self.baseline_metrics:
                ax1.axhline(y=self.baseline_metrics["mean_accuracy"], color='red', linestyle='--', 
                           label=f'Baseline: {self.baseline_metrics["mean_accuracy"]:.3f}')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('')
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        
        # Latency plot
        ax2 = fig.add_subplot(gs[0, 1])
        if "current_latency" in metrics_df.columns:
            # Convert to milliseconds
            latency_ms = metrics_df["current_latency"] * 1000
            latency_ms.plot(x=metrics_df["timestamp"], ax=ax2, color='green', marker='o', markersize=4)
            if self.baseline_metrics and "mean_latency" in self.baseline_metrics:
                ax2.axhline(y=self.baseline_metrics["mean_latency"] * 1000, color='red', linestyle='--', 
                           label=f'Baseline: {self.baseline_metrics["mean_latency"]*1000:.2f}ms')
        ax2.set_title('Model Latency')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_xlabel('')
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        
        # Confidence plot
        ax3 = fig.add_subplot(gs[1, 0])
        if "current_confidence" in metrics_df.columns:
            metrics_df.plot(x="timestamp", y="current_confidence", ax=ax3, color='purple', marker='o', markersize=4)
            if self.baseline_metrics and "mean_confidence" in self.baseline_metrics:
                ax3.axhline(y=self.baseline_metrics["mean_confidence"], color='red', linestyle='--', 
                           label=f'Baseline: {self.baseline_metrics["mean_confidence"]:.3f}')
        ax3.set_title('Model Confidence')
        ax3.set_ylabel('Confidence')
        ax3.set_xlabel('')
        ax3.legend()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        
        # Drift plot
        ax4 = fig.add_subplot(gs[1, 1])
        if "current_drift" in metrics_df.columns:
            metrics_df.plot(x="timestamp", y="current_drift", ax=ax4, color='orange', marker='o', markersize=4)
            ax4.axhline(y=0, color='red', linestyle='--', label='No Drift')
        ax4.set_title('Model Drift')
        ax4.set_ylabel('Drift Score')
        ax4.set_xlabel('')
        ax4.legend()
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        
        # Alerts plot
        ax5 = fig.add_subplot(gs[2, :])
        if not alerts_df.empty and "timestamp" in alerts_df.columns and "metric" in alerts_df.columns:
            # Create scatter plot with different colors for different metrics
            metrics = alerts_df["metric"].unique()
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            
            for i, metric in enumerate(metrics):
                metric_alerts = alerts_df[alerts_df["metric"] == metric]
                ax5.scatter(metric_alerts["timestamp"], [1] * len(metric_alerts), 
                           label=metric, color=colors[i % len(colors)], s=100, marker='o')
            
            # Add alert messages as annotations
            for _, alert in alerts_df.iterrows():
                ax5.annotate(alert["message"], 
                            (alert["timestamp"], 1),
                            xytext=(0, 10),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        ax5.set_title('Model Alerts')
        ax5.set_yticks([])
        ax5.set_xlabel('Time')
        if not alerts_df.empty:
            ax5.legend(title="Alert Type")
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save the dashboard
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"performance_dashboard_{timestamp}.png")
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance dashboard generated: {output_file}")
        
        return output_file
    
    def generate_html_dashboard(self, days: int = 7, output_file: str = None) -> str:
        """
        Generate HTML performance dashboard
        
        Args:
            days: Number of days of data to include
            output_file: Output file path
        
        Returns:
            Path to the generated dashboard file
        """
        
        logger.info(f"Generating HTML performance dashboard for the last {days} days...")
        
        # Load metrics data
        metrics_df = self._load_metrics_data(days)
        
        if metrics_df.empty:
            logger.warning("No metrics data available")
            return None
        
        # Load alerts data
        alerts_df = self._load_alerts_data(days)
        
        # Generate performance dashboard image
        dashboard_img = self.generate_performance_dashboard(days)
        
        # Calculate summary statistics
        current_metrics = metrics_df.iloc[-1] if not metrics_df.empty else {}
        
        # Current values
        current_accuracy = current_metrics.get("current_accuracy", "N/A")
        current_latency = current_metrics.get("current_latency", "N/A")
        current_confidence = current_metrics.get("current_confidence", "N/A")
        current_drift = current_metrics.get("current_drift", "N/A")
        
        # Changes from baseline
        accuracy_change = current_metrics.get("accuracy_change", "N/A")
        latency_change = current_metrics.get("latency_change", "N/A")
        confidence_change = current_metrics.get("confidence_change", "N/A")
        
        # Alert count
        alert_count = len(alerts_df) if not alerts_df.empty else 0
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MedChain AI Performance Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    background-color: #4a6fa5;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .header h1 {{
                    margin: 0;
                }}
                .status {{
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-weight: bold;
                }}
                .status-ok {{
                    background-color: #4caf50;
                }}
                .status-alert {{
                    background-color: #f44336;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .metric-card {{
                    background-color: white;
                    border-radius: 5px;
                    padding: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric-title {{
                    font-size: 16px;
                    color: #666;
                    margin-bottom: 10px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .metric-change {{
                    font-size: 14px;
                    display: flex;
                    align-items: center;
                }}
                .change-positive {{
                    color: #4caf50;
                }}
                .change-negative {{
                    color: #f44336;
                }}
                .change-neutral {{
                    color: #888;
                }}
                .dashboard-image {{
                    width: 100%;
                    height: auto;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .alerts-section {{
                    background-color: white;
                    border-radius: 5px;
                    padding: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .alert-item {{
                    padding: 10px;
                    border-bottom: 1px solid #eee;
                }}
                .alert-item:last-child {{
                    border-bottom: none;
                }}
                .alert-time {{
                    font-size: 12px;
                    color: #888;
                }}
                .alert-message {{
                    font-size: 14px;
                    margin-top: 5px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 20px;
                    color: #888;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>MedChain AI Performance Dashboard</h1>
                    <div class="status {'status-alert' if alert_count > 0 else 'status-ok'}">
                        {f"{alert_count} Alerts" if alert_count > 0 else "System OK"}
                    </div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Accuracy</div>
                        <div class="metric-value">{current_accuracy if current_accuracy != "N/A" else "N/A"}</div>
                        <div class="metric-change {'change-positive' if accuracy_change != 'N/A' and accuracy_change > 0 else 'change-negative' if accuracy_change != 'N/A' and accuracy_change < 0 else 'change-neutral'}">
                            {f"{accuracy_change*100:+.1f}%" if accuracy_change != "N/A" else "N/A"} from baseline
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Latency</div>
                        <div class="metric-value">{f"{current_latency*1000:.2f}ms" if current_latency != "N/A" else "N/A"}</div>
                        <div class="metric-change {'change-negative' if latency_change != 'N/A' and latency_change > 0 else 'change-positive' if latency_change != 'N/A' and latency_change < 0 else 'change-neutral'}">
                            {f"{latency_change*1000:+.2f}ms" if latency_change != "N/A" else "N/A"} from baseline
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Confidence</div>
                        <div class="metric-value">{current_confidence if current_confidence != "N/A" else "N/A"}</div>
                        <div class="metric-change {'change-positive' if confidence_change != 'N/A' and confidence_change > 0 else 'change-negative' if confidence_change != 'N/A' and confidence_change < 0 else 'change-neutral'}">
                            {f"{confidence_change*100:+.1f}%" if confidence_change != "N/A" else "N/A"} from baseline
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Drift Score</div>
                        <div class="metric-value">{current_drift if current_drift != "N/A" else "N/A"}</div>
                        <div class="metric-change">
                            {f"{'High' if current_drift > 0.2 else 'Medium' if current_drift > 0.1 else 'Low'} drift detected" if current_drift != "N/A" else "N/A"}
                        </div>
                    </div>
                </div>
                
                <img class="dashboard-image" src="{os.path.basename(dashboard_img)}" alt="Performance Dashboard">
                
                <div class="alerts-section">
                    <h2>Recent Alerts ({alert_count})</h2>
                    {"".join([f'''
                    <div class="alert-item">
                        <div class="alert-time">{alert["timestamp"]}</div>
                        <div class="alert-message">{alert["message"]}</div>
                    </div>
                    ''' for _, alert in alerts_df.iterrows()][:10]) if not alerts_df.empty else "<p>No alerts in the selected time period.</p>"}
                </div>
                
                <div class="footer">
                    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML dashboard
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"performance_dashboard_{timestamp}.html")
        
        with open(output_file, "w") as f:
            f.write(html)
        
        # Copy the dashboard image to the same directory
        if dashboard_img and os.path.exists(dashboard_img):
            import shutil
            dest_path = os.path.join(os.path.dirname(output_file), os.path.basename(dashboard_img))
            if dashboard_img != dest_path:  # Only copy if source and destination are different
                shutil.copy(dashboard_img, dest_path)
        
        logger.info(f"HTML performance dashboard generated: {output_file}")
        
        return output_file

def main():
    """Main function"""
    
    print("\n" + "="*80)
    print("📊 MEDCHAIN AI PERFORMANCE DASHBOARD GENERATOR")
    print("="*80 + "\n")
    
    # Create dashboard
    dashboard = PerformanceDashboard()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate MedChain AI performance dashboard")
    parser.add_argument("--days", type=int, default=7, help="Number of days of data to include")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--format", type=str, choices=["html", "png"], default="html", help="Output format")
    args = parser.parse_args()
    
    # Generate dashboard
    if args.format == "html":
        output_file = dashboard.generate_html_dashboard(days=args.days, output_file=args.output)
    else:
        output_file = dashboard.generate_performance_dashboard(days=args.days, output_file=args.output)
    
    if output_file:
        print(f"\nDashboard generated: {output_file}")
    else:
        print("\nFailed to generate dashboard. Check logs for details.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()