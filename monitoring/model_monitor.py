#!/usr/bin/env python3
"""
MedChain AI Model Monitoring System
Real-time monitoring and alerting for AI model performance
"""

import os
import sys
import time
import json
import logging
import numpy as np
import threading
import queue
import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque

# Add the ai_model directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai_model'))

try:
    from pretrained_medical_ai import MedicalAIInference, create_test_patients
    print("✅ Successfully imported medical AI modules")
except ImportError as e:
    print(f"❌ Failed to import medical AI: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "monitoring.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_monitor")

class ModelMonitor:
    """
    Real-time monitoring system for MedChain AI model
    """
    
    def __init__(self, data_dir: str = None, alert_threshold: float = 0.2):
        """
        Initialize the monitoring system
        
        Args:
            data_dir: Directory to store monitoring data
            alert_threshold: Threshold for alerting on metric changes
        """
        self.model = MedicalAIInference()
        
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        else:
            self.data_dir = data_dir
            
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create subdirectories
        self.metrics_dir = os.path.join(self.data_dir, "metrics")
        self.alerts_dir = os.path.join(self.data_dir, "alerts")
        self.logs_dir = os.path.join(self.data_dir, "logs")
        
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.alerts_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Monitoring parameters
        self.alert_threshold = alert_threshold
        self.monitoring_interval = 60  # seconds
        self.max_history = 1000  # maximum number of data points to keep
        
        # Test data
        self.test_patients = create_test_patients()
        
        # Metrics history
        self.metrics_history = {
            "accuracy": deque(maxlen=self.max_history),
            "latency": deque(maxlen=self.max_history),
            "confidence": deque(maxlen=self.max_history),
            "drift": deque(maxlen=self.max_history),
            "timestamp": deque(maxlen=self.max_history)
        }
        
        # Baseline metrics
        self.baseline_metrics = None
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Inference queue for async processing
        self.inference_queue = queue.Queue()
        self.inference_thread = None
        
        # Alert history
        self.alerts = []
        
        logger.info("Model monitoring system initialized")
    
    def establish_baseline(self) -> Dict:
        """Establish baseline metrics for the model"""
        
        logger.info("Establishing baseline metrics...")
        
        baseline = {
            "accuracy": [],
            "latency": [],
            "confidence": [],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Run inference on test patients
        for i, patient in enumerate(self.test_patients):
            logger.info(f"Processing baseline patient {i+1}/{len(self.test_patients)}")
            
            # Get expected diagnosis
            expected = self._get_expected_diagnosis(patient)
            
            # Measure inference time
            start_time = time.time()
            diagnosis = self.model.diagnose_rare_disease(patient)
            inference_time = time.time() - start_time
            
            # Check if prediction matches expected
            predicted = diagnosis["primary_diagnosis"]
            confidence = diagnosis["confidence"]
            
            is_correct = expected in predicted
            
            baseline["accuracy"].append(1 if is_correct else 0)
            baseline["latency"].append(inference_time)
            baseline["confidence"].append(confidence)
        
        # Calculate aggregate metrics
        baseline["mean_accuracy"] = np.mean(baseline["accuracy"])
        baseline["mean_latency"] = np.mean(baseline["latency"])
        baseline["mean_confidence"] = np.mean(baseline["confidence"])
        
        # Save baseline
        self.baseline_metrics = baseline
        
        # Save to file
        baseline_path = os.path.join(self.metrics_dir, "baseline.json")
        with open(baseline_path, "w") as f:
            # Convert deques to lists for serialization
            serializable_baseline = {
                k: list(v) if isinstance(v, deque) else v 
                for k, v in baseline.items()
            }
            json.dump(serializable_baseline, f, indent=2)
        
        logger.info(f"Baseline metrics established: accuracy={baseline['mean_accuracy']:.3f}, "
                   f"latency={baseline['mean_latency']*1000:.2f}ms, "
                   f"confidence={baseline['mean_confidence']:.3f}")
        
        return baseline
    
    def start_monitoring(self) -> None:
        """Start the monitoring system"""
        
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        logger.info("Starting model monitoring...")
        
        # Establish baseline if not already done
        if self.baseline_metrics is None:
            self.establish_baseline()
        
        # Start inference thread
        self.stop_event.clear()
        self.inference_thread = threading.Thread(target=self._inference_worker)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.is_monitoring = True
        
        logger.info("Model monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        
        if not self.is_monitoring:
            logger.warning("Monitoring is not running")
            return
        
        logger.info("Stopping model monitoring...")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.inference_thread:
            self.inference_thread.join(timeout=5)
        
        self.is_monitoring = False
        
        logger.info("Model monitoring stopped")
    
    def submit_inference(self, patient: Dict) -> None:
        """Submit a patient for inference"""
        
        if not self.is_monitoring:
            logger.warning("Monitoring is not running, inference not processed")
            return
        
        # Add to queue
        self.inference_queue.put(patient)
    
    def get_current_metrics(self) -> Dict:
        """Get current monitoring metrics"""
        
        if not self.metrics_history["timestamp"]:
            return {
                "status": "No metrics available",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Calculate recent metrics
        recent_accuracy = list(self.metrics_history["accuracy"])[-10:]
        recent_latency = list(self.metrics_history["latency"])[-10:]
        recent_confidence = list(self.metrics_history["confidence"])[-10:]
        recent_drift = list(self.metrics_history["drift"])[-10:]
        
        metrics = {
            "current_accuracy": np.mean(recent_accuracy) if recent_accuracy else None,
            "current_latency": np.mean(recent_latency) if recent_latency else None,
            "current_confidence": np.mean(recent_confidence) if recent_confidence else None,
            "current_drift": np.mean(recent_drift) if recent_drift else None,
            "baseline_accuracy": self.baseline_metrics["mean_accuracy"] if self.baseline_metrics else None,
            "baseline_latency": self.baseline_metrics["mean_latency"] if self.baseline_metrics else None,
            "baseline_confidence": self.baseline_metrics["mean_confidence"] if self.baseline_metrics else None,
            "accuracy_change": (np.mean(recent_accuracy) - self.baseline_metrics["mean_accuracy"]) 
                              if (self.baseline_metrics and recent_accuracy) else None,
            "latency_change": (np.mean(recent_latency) - self.baseline_metrics["mean_latency"]) 
                             if (self.baseline_metrics and recent_latency) else None,
            "confidence_change": (np.mean(recent_confidence) - self.baseline_metrics["mean_confidence"]) 
                                if (self.baseline_metrics and recent_confidence) else None,
            "timestamp": datetime.datetime.now().isoformat(),
            "alerts": len(self.alerts),
            "status": "OK" if not self.alerts else "ALERT"
        }
        
        return metrics
    
    def generate_dashboard(self) -> str:
        """Generate a dashboard HTML file with monitoring metrics"""
        
        dashboard_path = os.path.join(self.data_dir, "dashboard.html")
        
        # Get current metrics
        metrics = self.get_current_metrics()
        
        # Convert metrics history to lists
        accuracy_history = list(self.metrics_history["accuracy"])
        latency_history = list(self.metrics_history["latency"])
        confidence_history = list(self.metrics_history["confidence"])
        drift_history = list(self.metrics_history["drift"])
        timestamp_history = list(self.metrics_history["timestamp"])
        
        # Generate plots
        if timestamp_history:
            # Accuracy plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(accuracy_history)), accuracy_history, 'b-')
            if self.baseline_metrics:
                plt.axhline(y=self.baseline_metrics["mean_accuracy"], color='r', linestyle='--', 
                           label=f'Baseline: {self.baseline_metrics["mean_accuracy"]:.3f}')
            plt.title('Model Accuracy Over Time')
            plt.xlabel('Monitoring Interval')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            accuracy_plot_path = os.path.join(self.data_dir, "accuracy_plot.png")
            plt.savefig(accuracy_plot_path)
            plt.close()
            
            # Latency plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(latency_history)), [t*1000 for t in latency_history], 'g-')
            if self.baseline_metrics:
                plt.axhline(y=self.baseline_metrics["mean_latency"]*1000, color='r', linestyle='--', 
                           label=f'Baseline: {self.baseline_metrics["mean_latency"]*1000:.2f}ms')
            plt.title('Model Latency Over Time')
            plt.xlabel('Monitoring Interval')
            plt.ylabel('Latency (ms)')
            plt.legend()
            plt.grid(True)
            latency_plot_path = os.path.join(self.data_dir, "latency_plot.png")
            plt.savefig(latency_plot_path)
            plt.close()
            
            # Confidence plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(confidence_history)), confidence_history, 'm-')
            if self.baseline_metrics:
                plt.axhline(y=self.baseline_metrics["mean_confidence"], color='r', linestyle='--', 
                           label=f'Baseline: {self.baseline_metrics["mean_confidence"]:.3f}')
            plt.title('Model Confidence Over Time')
            plt.xlabel('Monitoring Interval')
            plt.ylabel('Confidence')
            plt.legend()
            plt.grid(True)
            confidence_plot_path = os.path.join(self.data_dir, "confidence_plot.png")
            plt.savefig(confidence_plot_path)
            plt.close()
            
            # Drift plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(drift_history)), drift_history, 'y-')
            plt.axhline(y=0, color='r', linestyle='--', label='No Drift')
            plt.title('Model Drift Over Time')
            plt.xlabel('Monitoring Interval')
            plt.ylabel('Drift Score')
            plt.legend()
            plt.grid(True)
            drift_plot_path = os.path.join(self.data_dir, "drift_plot.png")
            plt.savefig(drift_plot_path)
            plt.close()
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MedChain AI Model Monitoring Dashboard</title>
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
                    grid-template-columns: repeat(3, 1fr);
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
                .plots-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .plot-card {{
                    background-color: white;
                    border-radius: 5px;
                    padding: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .plot-title {{
                    font-size: 18px;
                    margin-bottom: 15px;
                    color: #333;
                }}
                .plot-image {{
                    width: 100%;
                    height: auto;
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
                    <h1>MedChain AI Model Monitoring Dashboard</h1>
                    <div class="status {{'status-ok' if metrics['status'] == 'OK' else 'status-alert'}}">
                        {metrics['status']}
                    </div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Accuracy</div>
                        <div class="metric-value">{f"{metrics.get('current_accuracy'):.3f}" if metrics.get('current_accuracy') is not None else 'N/A'}</div>
                        <div class="metric-change {{'change-positive' if metrics.get('accuracy_change', 0) > 0 else 'change-negative' if metrics.get('accuracy_change', 0) < 0 else 'change-neutral'}}">
                            {f"{metrics['accuracy_change']*100:+.1f}%" if metrics.get('accuracy_change') is not None else 'N/A'} from baseline
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Latency</div>
                        <div class="metric-value">{f"{metrics.get('current_latency')*1000:.2f}ms" if metrics.get('current_latency') is not None else 'N/A'}</div>
                        <div class="metric-change {{'change-negative' if metrics.get('latency_change', 0) > 0 else 'change-positive' if metrics.get('latency_change', 0) < 0 else 'change-neutral'}}">
                            {f"{metrics['latency_change']*1000:+.2f}ms" if metrics.get('latency_change') is not None else 'N/A'} from baseline
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Confidence</div>
                        <div class="metric-value">{f"{metrics.get('current_confidence'):.3f}" if metrics.get('current_confidence') is not None else 'N/A'}</div>
                        <div class="metric-change {{'change-positive' if metrics.get('confidence_change', 0) > 0 else 'change-negative' if metrics.get('confidence_change', 0) < 0 else 'change-neutral'}}">
                            {f"{metrics['confidence_change']*100:+.1f}%" if metrics.get('confidence_change') is not None else 'N/A'} from baseline
                        </div>
                    </div>
                </div>
                
                <div class="plots-grid">
                    <div class="plot-card">
                        <div class="plot-title">Accuracy Over Time</div>
                        <img class="plot-image" src="accuracy_plot.png" alt="Accuracy Plot">
                    </div>
                    
                    <div class="plot-card">
                        <div class="plot-title">Latency Over Time</div>
                        <img class="plot-image" src="latency_plot.png" alt="Latency Plot">
                    </div>
                    
                    <div class="plot-card">
                        <div class="plot-title">Confidence Over Time</div>
                        <img class="plot-image" src="confidence_plot.png" alt="Confidence Plot">
                    </div>
                    
                    <div class="plot-card">
                        <div class="plot-title">Drift Over Time</div>
                        <img class="plot-image" src="drift_plot.png" alt="Drift Plot">
                    </div>
                </div>
                
                <div class="alerts-section">
                    <h2>Recent Alerts ({len(self.alerts)})</h2>
                    {''.join([f"""
                    <div class="alert-item">
                        <div class="alert-time">{alert['timestamp']}</div>
                        <div class="alert-message">{alert['message']}</div>
                    </div>
                    """ for alert in self.alerts[-10:]])}
                </div>
                
                <div class="footer">
                    Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(dashboard_path, "w") as f:
            f.write(html)
        
        logger.info(f"Dashboard generated: {dashboard_path}")
        
        return dashboard_path
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        
        logger.info("Monitoring loop started")
        
        while not self.stop_event.is_set():
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Check for drift
                self._check_for_drift()
                
                # Generate dashboard
                self.generate_dashboard()
                
                # Save metrics
                self._save_metrics()
                
                # Wait for next interval
                self.stop_event.wait(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _inference_worker(self) -> None:
        """Worker thread for processing inference requests"""
        
        logger.info("Inference worker started")
        
        while not self.stop_event.is_set():
            try:
                # Get patient from queue with timeout
                try:
                    patient = self.inference_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process inference
                start_time = time.time()
                diagnosis = self.model.diagnose_rare_disease(patient)
                inference_time = time.time() - start_time
                
                # Log inference
                logger.info(f"Inference processed: {diagnosis['primary_diagnosis']} "
                           f"(confidence: {diagnosis['confidence']:.3f}, "
                           f"time: {inference_time*1000:.2f}ms)")
                
                # Mark task as done
                self.inference_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in inference worker: {e}")
    
    def _collect_metrics(self) -> None:
        """Collect current model metrics"""
        
        logger.info("Collecting metrics...")
        
        # Select a random test patient
        patient = np.random.choice(self.test_patients)
        
        # Get expected diagnosis
        expected = self._get_expected_diagnosis(patient)
        
        # Measure inference time
        start_time = time.time()
        diagnosis = self.model.diagnose_rare_disease(patient)
        inference_time = time.time() - start_time
        
        # Check if prediction matches expected
        predicted = diagnosis["primary_diagnosis"]
        confidence = diagnosis["confidence"]
        
        is_correct = expected in predicted
        
        # Calculate drift from baseline
        if self.baseline_metrics:
            accuracy_drift = 1 if is_correct else 0 - self.baseline_metrics["mean_accuracy"]
            latency_drift = inference_time - self.baseline_metrics["mean_latency"]
            confidence_drift = confidence - self.baseline_metrics["mean_confidence"]
            
            # Combined drift score (weighted average)
            drift_score = (0.5 * abs(accuracy_drift) + 
                          0.3 * abs(latency_drift) / self.baseline_metrics["mean_latency"] + 
                          0.2 * abs(confidence_drift))
        else:
            drift_score = 0
        
        # Add to history
        timestamp = datetime.datetime.now().isoformat()
        self.metrics_history["accuracy"].append(1 if is_correct else 0)
        self.metrics_history["latency"].append(inference_time)
        self.metrics_history["confidence"].append(confidence)
        self.metrics_history["drift"].append(drift_score)
        self.metrics_history["timestamp"].append(timestamp)
        
        logger.info(f"Metrics collected: accuracy={1 if is_correct else 0}, "
                   f"latency={inference_time*1000:.2f}ms, "
                   f"confidence={confidence:.3f}, "
                   f"drift={drift_score:.3f}")
    
    def _check_for_drift(self) -> None:
        """Check for model drift and generate alerts"""
        
        if not self.baseline_metrics or len(self.metrics_history["accuracy"]) < 5:
            return
        
        # Get recent metrics
        recent_accuracy = list(self.metrics_history["accuracy"])[-5:]
        recent_latency = list(self.metrics_history["latency"])[-5:]
        recent_confidence = list(self.metrics_history["confidence"])[-5:]
        
        # Calculate changes from baseline
        accuracy_change = np.mean(recent_accuracy) - self.baseline_metrics["mean_accuracy"]
        latency_change = np.mean(recent_latency) - self.baseline_metrics["mean_latency"]
        confidence_change = np.mean(recent_confidence) - self.baseline_metrics["mean_confidence"]
        
        # Check for significant changes
        if abs(accuracy_change) > self.alert_threshold:
            direction = "decreased" if accuracy_change < 0 else "increased"
            message = f"Model accuracy has {direction} by {abs(accuracy_change)*100:.1f}%"
            self._generate_alert(message, "accuracy", accuracy_change)
        
        if abs(latency_change) / self.baseline_metrics["mean_latency"] > self.alert_threshold:
            direction = "increased" if latency_change > 0 else "decreased"
            message = f"Model latency has {direction} by {abs(latency_change)*1000:.2f}ms"
            self._generate_alert(message, "latency", latency_change)
        
        if abs(confidence_change) > self.alert_threshold:
            direction = "decreased" if confidence_change < 0 else "increased"
            message = f"Model confidence has {direction} by {abs(confidence_change)*100:.1f}%"
            self._generate_alert(message, "confidence", confidence_change)
    
    def _generate_alert(self, message: str, metric: str, change: float) -> None:
        """Generate an alert"""
        
        timestamp = datetime.datetime.now().isoformat()
        
        alert = {
            "timestamp": timestamp,
            "message": message,
            "metric": metric,
            "change": change
        }
        
        self.alerts.append(alert)
        
        # Save alert to file
        alert_path = os.path.join(self.alerts_dir, f"alert_{timestamp.replace(':', '-')}.json")
        with open(alert_path, "w") as f:
            json.dump(alert, f, indent=2)
        
        logger.warning(f"ALERT: {message}")
    
    def _save_metrics(self) -> None:
        """Save current metrics to file"""
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(self.metrics_dir, f"metrics_{timestamp}.json")
        
        # Get current metrics
        metrics = self.get_current_metrics()
        
        # Save to file
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    
    def _get_expected_diagnosis(self, patient: Dict) -> str:
        """Get expected diagnosis based on patient symptoms"""
        
        # Simple mapping based on symptoms
        symptoms = patient.get("symptoms", [])
        
        if any(s in ["chorea", "involuntary movements", "cognitive decline"] for s in symptoms):
            return "Huntington Disease"
        elif any(s in ["chronic cough", "thick mucus", "recurrent lung infections"] for s in symptoms):
            return "Cystic Fibrosis"
        elif any(s in ["muscle weakness", "double vision", "drooping eyelids"] for s in symptoms):
            return "Myasthenia Gravis"
        elif any(s in ["muscle weakness", "muscle atrophy", "fasciculations"] for s in symptoms):
            return "Amyotrophic Lateral Sclerosis"
        elif any(s in ["liver problems", "neurological symptoms", "tremor"] for s in symptoms):
            return "Wilson Disease"
        else:
            return "Unknown"

class ModelMonitorCLI:
    """Command-line interface for the model monitor"""
    
    def __init__(self):
        self.monitor = ModelMonitor()
    
    def run(self):
        """Run the CLI"""
        
        print("\n" + "="*80)
        print("🔍 MEDCHAIN AI MODEL MONITORING SYSTEM")
        print("="*80 + "\n")
        
        while True:
            print("\nOptions:")
            print("1. Establish baseline metrics")
            print("2. Start monitoring")
            print("3. Stop monitoring")
            print("4. View current metrics")
            print("5. Generate dashboard")
            print("6. Submit test inference")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ")
            
            if choice == "1":
                self.monitor.establish_baseline()
                print("Baseline metrics established")
            
            elif choice == "2":
                self.monitor.start_monitoring()
                print("Monitoring started")
            
            elif choice == "3":
                self.monitor.stop_monitoring()
                print("Monitoring stopped")
            
            elif choice == "4":
                metrics = self.monitor.get_current_metrics()
                print("\nCurrent Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            
            elif choice == "5":
                dashboard_path = self.monitor.generate_dashboard()
                print(f"Dashboard generated: {dashboard_path}")
            
            elif choice == "6":
                if not self.monitor.is_monitoring:
                    print("Monitoring must be started first")
                    continue
                
                # Select a random test patient
                patient = np.random.choice(self.monitor.test_patients)
                self.monitor.submit_inference(patient)
                print("Test inference submitted")
            
            elif choice == "7":
                if self.monitor.is_monitoring:
                    self.monitor.stop_monitoring()
                print("Exiting...")
                break
            
            else:
                print("Invalid choice, please try again")

if __name__ == "__main__":
    # Run the CLI
    cli = ModelMonitorCLI()
    cli.run()