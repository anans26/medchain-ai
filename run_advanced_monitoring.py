#!/usr/bin/env python3
"""
MedChain AI Advanced Monitoring System
Run the test suite and monitoring system
"""

import os
import sys
import argparse
import logging
import time
import threading
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advanced_monitoring")

def run_test_suite():
    """Run the advanced test suite"""
    
    logger.info("Running advanced test suite...")
    
    try:
        from test_suite.advanced_test_suite import AdvancedTestSuite
        
        # Create test suite
        test_suite = AdvancedTestSuite()
        
        # Run all tests
        results = test_suite.run_all_tests()
        
        logger.info(f"Test suite completed. Results saved to: {test_suite.output_dir}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error running test suite: {e}")
        return None

def run_model_monitor(duration: int = 0):
    """
    Run the model monitoring system
    
    Args:
        duration: Duration to run in seconds (0 for indefinite)
    """
    
    logger.info(f"Running model monitor for {duration if duration > 0 else 'indefinite'} seconds...")
    
    try:
        from monitoring.model_monitor import ModelMonitor
        
        # Create monitor
        monitor = ModelMonitor()
        
        # Establish baseline
        monitor.establish_baseline()
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Run for specified duration
        if duration > 0:
            time.sleep(duration)
            monitor.stop_monitoring()
        else:
            # Run indefinitely (until interrupted)
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Monitoring interrupted by user")
                monitor.stop_monitoring()
        
        logger.info("Model monitoring completed")
        
    except Exception as e:
        logger.error(f"Error running model monitor: {e}")

def run_alert_system(duration: int = 0):
    """
    Run the alert system
    
    Args:
        duration: Duration to run in seconds (0 for indefinite)
    """
    
    logger.info(f"Running alert system for {duration if duration > 0 else 'indefinite'} seconds...")
    
    try:
        from monitoring.alerts.alert_system import AlertSystem
        
        # Create alert system
        alert_system = AlertSystem()
        
        # Start alert system
        alert_system.start()
        
        # Run for specified duration
        if duration > 0:
            time.sleep(duration)
            alert_system.stop()
        else:
            # Run indefinitely (until interrupted)
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Alert system interrupted by user")
                alert_system.stop()
        
        logger.info("Alert system completed")
        
    except Exception as e:
        logger.error(f"Error running alert system: {e}")

def run_performance_dashboard():
    """Generate the performance dashboard"""
    
    logger.info("Generating performance dashboard...")
    
    try:
        from monitoring.dashboards.performance_dashboard import PerformanceDashboard
        
        # Create dashboard
        dashboard = PerformanceDashboard()
        
        # Generate dashboard
        output_file = dashboard.generate_html_dashboard()
        
        logger.info(f"Performance dashboard generated: {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error generating performance dashboard: {e}")
        return None

def run_all(duration: int = 3600):
    """
    Run all components
    
    Args:
        duration: Duration to run monitoring in seconds
    """
    
    logger.info("Running all components...")
    
    # Run test suite
    test_results = run_test_suite()
    
    # Start model monitor and alert system in separate threads
    monitor_thread = threading.Thread(target=run_model_monitor, args=(duration,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    alert_thread = threading.Thread(target=run_alert_system, args=(duration,))
    alert_thread.daemon = True
    alert_thread.start()
    
    # Wait for monitoring to complete
    monitor_thread.join()
    alert_thread.join()
    
    # Generate dashboard
    dashboard_file = run_performance_dashboard()
    
    logger.info("All components completed")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="MedChain AI Advanced Monitoring System")
    parser.add_argument("--component", type=str, choices=["test", "monitor", "alert", "dashboard", "all"], 
                       default="all", help="Component to run")
    parser.add_argument("--duration", type=int, default=3600, 
                       help="Duration to run monitoring in seconds (0 for indefinite)")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 MEDCHAIN AI ADVANCED MONITORING SYSTEM")
    print("="*80 + "\n")
    
    if args.component == "test":
        run_test_suite()
    elif args.component == "monitor":
        run_model_monitor(args.duration)
    elif args.component == "alert":
        run_alert_system(args.duration)
    elif args.component == "dashboard":
        run_performance_dashboard()
    else:  # all
        run_all(args.duration)
    
    print("\n" + "="*80)
    print("✅ MEDCHAIN AI ADVANCED MONITORING COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()