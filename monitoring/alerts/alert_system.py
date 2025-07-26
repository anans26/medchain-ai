#!/usr/bin/env python3
"""
MedChain AI Alert System
Monitors model performance and sends alerts when issues are detected
"""

import os
import sys
import json
import logging
import time
import datetime
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "alerts.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("alert_system")

class AlertSystem:
    """
    Alert system for MedChain AI model monitoring
    """
    
    def __init__(self, data_dir: str = None, config_file: str = None):
        """
        Initialize the alert system
        
        Args:
            data_dir: Directory containing monitoring data
            config_file: Path to alert configuration file
        """
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        else:
            self.data_dir = data_dir
            
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            logger.warning(f"Data directory not found, created: {self.data_dir}")
        
        # Alerts directory
        self.alerts_dir = os.path.join(self.data_dir, "alerts")
        if not os.path.exists(self.alerts_dir):
            os.makedirs(self.alerts_dir, exist_ok=True)
        
        # Load configuration
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), "alert_config.json")
        
        self.config = self._load_config(config_file)
        
        # Initialize model monitor
        self.monitor = ModelMonitor(data_dir=self.data_dir)
        
        # Alert state
        self.last_alert_time = {}
        self.alert_count = 0
        self.processed_alerts = set()
        
        # Alert thread
        self.is_running = False
        self.stop_event = threading.Event()
        self.alert_thread = None
        
        logger.info("Alert system initialized")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load alert configuration from file"""
        
        default_config = {
            "alert_thresholds": {
                "accuracy": 0.1,
                "latency": 0.2,
                "confidence": 0.1,
                "drift": 0.2
            },
            "alert_cooldown": 3600,  # seconds
            "check_interval": 300,  # seconds
            "email": {
                "enabled": False,
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "username": "alerts@example.com",
                "password": "password",
                "from_address": "alerts@example.com",
                "to_addresses": ["admin@example.com"]
            },
            "slack": {
                "enabled": False,
                "webhook_url": "https://hooks.slack.com/services/xxx/yyy/zzz"
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded alert configuration from {config_file}")
                
                # Merge with default config to ensure all fields are present
                merged_config = default_config.copy()
                merged_config.update(config)
                return merged_config
            except Exception as e:
                logger.error(f"Error loading alert configuration: {e}")
        
        logger.warning(f"Alert configuration file not found: {config_file}")
        logger.info("Using default alert configuration")
        
        # Save default config
        try:
            with open(config_file, "w") as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Saved default alert configuration to {config_file}")
        except Exception as e:
            logger.error(f"Error saving default alert configuration: {e}")
        
        return default_config
    
    def start(self) -> None:
        """Start the alert system"""
        
        if self.is_running:
            logger.warning("Alert system is already running")
            return
        
        logger.info("Starting alert system...")
        
        # Start model monitor if not already running
        if not self.monitor.is_monitoring:
            self.monitor.start_monitoring()
        
        # Start alert thread
        self.stop_event.clear()
        self.alert_thread = threading.Thread(target=self._alert_loop)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        self.is_running = True
        
        logger.info("Alert system started")
    
    def stop(self) -> None:
        """Stop the alert system"""
        
        if not self.is_running:
            logger.warning("Alert system is not running")
            return
        
        logger.info("Stopping alert system...")
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        
        self.is_running = False
        
        logger.info("Alert system stopped")
    
    def _alert_loop(self) -> None:
        """Main alert loop"""
        
        logger.info("Alert loop started")
        
        while not self.stop_event.is_set():
            try:
                # Check for new alerts
                self._check_alerts()
                
                # Wait for next check
                self.stop_event.wait(self.config["check_interval"])
                
            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _check_alerts(self) -> None:
        """Check for new alerts"""
        
        logger.info("Checking for new alerts...")
        
        # Get all alert files
        alert_files = glob.glob(os.path.join(self.alerts_dir, "alert_*.json"))
        
        # Process new alerts
        for file_path in alert_files:
            if file_path in self.processed_alerts:
                continue
            
            try:
                with open(file_path, "r") as f:
                    alert = json.load(f)
                
                # Process alert
                self._process_alert(alert)
                
                # Mark as processed
                self.processed_alerts.add(file_path)
                
            except Exception as e:
                logger.error(f"Error processing alert from {file_path}: {e}")
    
    def _process_alert(self, alert: Dict) -> None:
        """Process an alert"""
        
        logger.info(f"Processing alert: {alert['message']}")
        
        # Check cooldown
        metric = alert.get("metric", "unknown")
        current_time = time.time()
        
        if metric in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[metric]
            if time_since_last < self.config["alert_cooldown"]:
                logger.info(f"Alert for {metric} is in cooldown ({time_since_last:.0f}s < {self.config['alert_cooldown']}s)")
                return
        
        # Update last alert time
        self.last_alert_time[metric] = current_time
        
        # Increment alert count
        self.alert_count += 1
        
        # Send notifications
        self._send_email_alert(alert)
        self._send_slack_alert(alert)
        
        logger.info(f"Alert processed: {alert['message']}")
    
    def _send_email_alert(self, alert: Dict) -> None:
        """Send email alert"""
        
        if not self.config["email"]["enabled"]:
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.config["email"]["from_address"]
            msg["To"] = ", ".join(self.config["email"]["to_addresses"])
            msg["Subject"] = f"MedChain AI Alert: {alert['metric']} Issue Detected"
            
            # Create message body
            body = f"""
            <html>
            <body>
                <h2>MedChain AI Model Alert</h2>
                <p><strong>Alert:</strong> {alert['message']}</p>
                <p><strong>Timestamp:</strong> {alert['timestamp']}</p>
                <p><strong>Metric:</strong> {alert['metric']}</p>
                <p><strong>Change:</strong> {alert['change']}</p>
                <hr>
                <p>Please check the monitoring dashboard for more details.</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, "html"))
            
            # Connect to SMTP server
            server = smtplib.SMTP(self.config["email"]["smtp_server"], self.config["email"]["smtp_port"])
            server.starttls()
            server.login(self.config["email"]["username"], self.config["email"]["password"])
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent to {', '.join(self.config['email']['to_addresses'])}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _send_slack_alert(self, alert: Dict) -> None:
        """Send Slack alert"""
        
        if not self.config["slack"]["enabled"]:
            return
        
        try:
            import requests
            
            # Create message
            message = {
                "text": f"*MedChain AI Model Alert*",
                "attachments": [
                    {
                        "color": "#ff0000",
                        "fields": [
                            {
                                "title": "Alert",
                                "value": alert["message"],
                                "short": False
                            },
                            {
                                "title": "Metric",
                                "value": alert["metric"],
                                "short": True
                            },
                            {
                                "title": "Change",
                                "value": str(alert["change"]),
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": alert["timestamp"],
                                "short": False
                            }
                        ]
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(
                self.config["slack"]["webhook_url"],
                json=message,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info("Slack alert sent successfully")
            else:
                logger.error(f"Error sending Slack alert: {response.status_code} {response.text}")
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    def generate_alert(self, message: str, metric: str, change: float) -> None:
        """Manually generate an alert"""
        
        timestamp = datetime.datetime.now().isoformat()
        
        alert = {
            "timestamp": timestamp,
            "message": message,
            "metric": metric,
            "change": change
        }
        
        # Save alert to file
        alert_path = os.path.join(self.alerts_dir, f"alert_{timestamp.replace(':', '-')}.json")
        with open(alert_path, "w") as f:
            json.dump(alert, f, indent=2)
        
        logger.warning(f"ALERT GENERATED: {message}")
        
        # Process alert immediately
        self._process_alert(alert)

class AlertSystemCLI:
    """Command-line interface for the alert system"""
    
    def __init__(self):
        self.alert_system = AlertSystem()
    
    def run(self):
        """Run the CLI"""
        
        print("\n" + "="*80)
        print("🚨 MEDCHAIN AI ALERT SYSTEM")
        print("="*80 + "\n")
        
        while True:
            print("\nOptions:")
            print("1. Start alert system")
            print("2. Stop alert system")
            print("3. Generate test alert")
            print("4. View alert configuration")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == "1":
                self.alert_system.start()
                print("Alert system started")
            
            elif choice == "2":
                self.alert_system.stop()
                print("Alert system stopped")
            
            elif choice == "3":
                metric = input("Enter metric (accuracy, latency, confidence, drift): ")
                message = input("Enter alert message: ")
                change = float(input("Enter change value: "))
                
                self.alert_system.generate_alert(message, metric, change)
                print("Test alert generated")
            
            elif choice == "4":
                print("\nAlert Configuration:")
                for key, value in self.alert_system.config.items():
                    print(f"  {key}: {value}")
            
            elif choice == "5":
                if self.alert_system.is_running:
                    self.alert_system.stop()
                print("Exiting...")
                break
            
            else:
                print("Invalid choice, please try again")

if __name__ == "__main__":
    # Run the CLI
    cli = AlertSystemCLI()
    cli.run()