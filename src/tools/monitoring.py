#!/usr/bin/env python3
"""
Monitoring and metrics for MCP Think Tank
Provides logging, performance tracking, and usage analytics
"""
import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, Callable, TypeVar, List

import threading
import atexit

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("mcp-think-tank.monitoring")

T = TypeVar("T")


class MetricType(str, Enum):
    """Types of metrics to track"""
    COUNTER = "counter"    # Cumulative count (e.g., number of tool calls)
    GAUGE = "gauge"        # Current value (e.g., memory usage)
    HISTOGRAM = "histogram"  # Distribution of values (e.g., response times)
    TIMER = "timer"        # Duration of operations

class MetricsCollector:
    """
    Collects and logs usage metrics for MCP Think Tank
    """
    
    def __init__(self, log_file_path: Optional[str] = None, 
                 flush_interval: int = 60, 
                 enable_file_logging: bool = True):
        """
        Initialize the metrics collector
        
        Args:
            log_file_path: Path to the metrics log file
            flush_interval: How often (in seconds) to flush metrics to disk
            enable_file_logging: Whether to log metrics to file
        """
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.log_file_path = log_file_path or os.path.expanduser("~/.mcp-think-tank/metrics.jsonl")
        self.enable_file_logging = enable_file_logging
        self.flush_interval = flush_interval
        self.lock = threading.RLock()
        self.last_flush_time = time.time()
        
        # Ensure the directory exists
        if self.enable_file_logging:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        
        # Register exit handler to flush metrics
        atexit.register(self.flush_metrics)
        
        # Start background thread for periodic flushing
        if self.enable_file_logging and self.flush_interval > 0:
            self.flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
            self.flush_thread.start()
    
    def _periodic_flush(self):
        """Background thread for periodic flushing of metrics"""
        while True:
            time.sleep(self.flush_interval)
            try:
                self.flush_metrics()
            except Exception as e:
                logger.error(f"Failed to flush metrics: {e}")
    
    def track(self, name: str, value: Any, metric_type: MetricType = MetricType.COUNTER, 
              labels: Optional[Dict[str, str]] = None) -> None:
        """
        Track a metric
        
        Args:
            name: Name of the metric
            value: Value to track
            metric_type: Type of metric (counter, gauge, histogram, timer)
            labels: Optional labels to associate with the metric
        """
        with self.lock:
            # Initialize metric if it doesn't exist
            if name not in self.metrics:
                self.metrics[name] = {
                    "type": metric_type,
                    "values": [],
                    "last_value": None,
                    "count": 0,
                    "sum": 0,
                    "min": None,
                    "max": None
                }
            
            metric = self.metrics[name]
            
            # Update metric based on type
            if metric_type == MetricType.COUNTER:
                metric["count"] += value
                metric["last_value"] = value
            elif metric_type == MetricType.GAUGE:
                metric["last_value"] = value
            elif metric_type == MetricType.HISTOGRAM or metric_type == MetricType.TIMER:
                if not isinstance(metric["values"], list):
                    metric["values"] = []
                metric["values"].append(value)
                metric["count"] += 1
                metric["sum"] += value
                
                # Update min/max
                if metric["min"] is None or value < metric["min"]:
                    metric["min"] = value
                if metric["max"] is None or value > metric["max"]:
                    metric["max"] = value
                
                # Keep only last 1000 values for histograms to save memory
                if len(metric["values"]) > 1000:
                    metric["values"] = metric["values"][-1000:]
            
            # Flush if it's been more than flush_interval since last flush
            current_time = time.time()
            if self.enable_file_logging and current_time - self.last_flush_time >= self.flush_interval:
                self.flush_metrics()
    
    def increment(self, name: str, amount: int = 1, 
                 labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric
        
        Args:
            name: Name of the metric
            amount: Amount to increment by
            labels: Optional labels to associate with the metric
        """
        self.track(name, amount, MetricType.COUNTER, labels)
    
    def gauge(self, name: str, value: Any, 
             labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric
        
        Args:
            name: Name of the metric
            value: Value to set
            labels: Optional labels to associate with the metric
        """
        self.track(name, value, MetricType.GAUGE, labels)
    
    def histogram(self, name: str, value: float, 
                 labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a value in a histogram metric
        
        Args:
            name: Name of the metric
            value: Value to record
            labels: Optional labels to associate with the metric
        """
        self.track(name, value, MetricType.HISTOGRAM, labels)
    
    def time_this(self, name: str, labels: Optional[Dict[str, str]] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Timer decorator for measuring function execution time
        
        Args:
            name: Name of the timer metric
            labels: Optional labels to associate with the metric
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args: Any, **kwargs: Any) -> T:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.track(name, execution_time, MetricType.TIMER, labels)
                return result
            return wrapper
        return decorator
    
    def flush_metrics(self) -> None:
        """Flush metrics to log file"""
        if not self.enable_file_logging:
            return
            
        with self.lock:
            try:
                # Create a snapshot of current metrics
                timestamp = datetime.now().isoformat()
                metrics_snapshot = {
                    "timestamp": timestamp,
                    "metrics": {}
                }
                
                for name, metric in self.metrics.items():
                    # Skip metrics with no data
                    if (metric["type"] == MetricType.COUNTER and metric["count"] == 0) or \
                       (metric["type"] == MetricType.GAUGE and metric["last_value"] is None) or \
                       (metric["type"] == MetricType.HISTOGRAM and not metric["values"]) or \
                       (metric["type"] == MetricType.TIMER and not metric["values"]):
                        continue
                    
                    # Calculate statistics for histograms and timers
                    if metric["type"] in [MetricType.HISTOGRAM, MetricType.TIMER] and metric["count"] > 0:
                        avg = metric["sum"] / metric["count"]
                        
                        # Calculate percentiles if we have enough data
                        percentiles = {}
                        if metric["values"]:
                            sorted_values = sorted(metric["values"])
                            for p in [50, 90, 95, 99]:
                                idx = min(int(len(sorted_values) * p / 100), len(sorted_values) - 1)
                                percentiles[f"p{p}"] = sorted_values[idx]
                        
                        metrics_snapshot["metrics"][name] = {
                            "type": metric["type"],
                            "count": metric["count"],
                            "sum": metric["sum"],
                            "avg": avg,
                            "min": metric["min"],
                            "max": metric["max"],
                            **percentiles
                        }
                    elif metric["type"] == MetricType.COUNTER:
                        metrics_snapshot["metrics"][name] = {
                            "type": metric["type"],
                            "count": metric["count"],
                            "last_value": metric["last_value"]
                        }
                    elif metric["type"] == MetricType.GAUGE:
                        metrics_snapshot["metrics"][name] = {
                            "type": metric["type"],
                            "value": metric["last_value"]
                        }
                
                # Append to log file
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(metrics_snapshot) + '\n')
                
                self.last_flush_time = time.time()
                logger.debug(f"Flushed metrics to {self.log_file_path}")
            except Exception as e:
                logger.error(f"Failed to flush metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics
        
        Returns:
            Dictionary with current metrics
        """
        with self.lock:
            result = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {}
            }
            
            for name, metric in self.metrics.items():
                if metric["type"] in [MetricType.HISTOGRAM, MetricType.TIMER] and metric["count"] > 0:
                    avg = metric["sum"] / metric["count"]
                    
                    # Calculate percentiles if we have enough data
                    percentiles = {}
                    if metric["values"]:
                        sorted_values = sorted(metric["values"])
                        for p in [50, 90, 95, 99]:
                            idx = min(int(len(sorted_values) * p / 100), len(sorted_values) - 1)
                            percentiles[f"p{p}"] = sorted_values[idx]
                    
                    result["metrics"][name] = {
                        "type": metric["type"],
                        "count": metric["count"],
                        "sum": metric["sum"],
                        "avg": avg,
                        "min": metric["min"],
                        "max": metric["max"],
                        **percentiles
                    }
                elif metric["type"] == MetricType.COUNTER:
                    result["metrics"][name] = {
                        "type": metric["type"],
                        "count": metric["count"],
                        "last_value": metric["last_value"]
                    }
                elif metric["type"] == MetricType.GAUGE:
                    result["metrics"][name] = {
                        "type": metric["type"],
                        "value": metric["last_value"]
                    }
            
            return result
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        with self.lock:
            self.metrics = {}


# Create global metrics collector
metrics = MetricsCollector()


class PerformanceTracker:
    """
    Tracks performance of operations
    """
    
    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Initialize performance tracker
        
        Args:
            name: Name of the operation
            labels: Optional labels to associate with the operation
        """
        self.name = name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        """Start tracking performance"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking performance and record metrics"""
        if self.start_time is not None:
            execution_time = time.time() - self.start_time
            metrics.track(f"{self.name}_time", execution_time, MetricType.TIMER, self.labels)
            
            # Also track success/failure
            if exc_type is not None:
                metrics.increment(f"{self.name}_errors", 1, self.labels)
            else:
                metrics.increment(f"{self.name}_success", 1, self.labels)
            
            # Reset start time
            self.start_time = None


def track_tool_call(tool_name: str, success: bool, execution_time: float) -> None:
    """
    Track a tool call
    
    Args:
        tool_name: Name of the tool
        success: Whether the call was successful
        execution_time: Time taken to execute the tool
    """
    # Track call count
    metrics.increment("tool_calls_total", 1, {"tool": tool_name})
    
    # Track success/failure
    if success:
        metrics.increment("tool_calls_success", 1, {"tool": tool_name})
    else:
        metrics.increment("tool_calls_errors", 1, {"tool": tool_name})
    
    # Track execution time
    metrics.track(f"tool_calls_time", execution_time, MetricType.TIMER, {"tool": tool_name})


def get_performance_metrics() -> Dict[str, Any]:
    """
    Get all performance metrics
    
    Returns:
        Dictionary with performance metrics
    """
    return metrics.get_metrics()


def track_memory_usage(tool_name: Optional[str] = None) -> None:
    """
    Track current memory usage
    
    Args:
        tool_name: Optional tool name to associate with memory usage
    """
    if psutil is None:
        logger.warning("psutil not installed, cannot track memory usage")
        return

    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Track as gauge (current value)
        labels = {"tool": tool_name} if tool_name else None
        metrics.gauge("memory_rss_bytes", memory_info.rss, labels)
        metrics.gauge("memory_vms_bytes", memory_info.vms, labels)
    except Exception as e:
        logger.error(f"Failed to track memory usage: {e}")


def track_file_size(file_path: str, name: str) -> None:
    """
    Track size of a file
    
    Args:
        file_path: Path to the file
        name: Name to use for the metric
    """
    try:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            metrics.gauge(f"{name}_size_bytes", size)
    except Exception as e:
        logger.error(f"Failed to track file size: {e}") 