"""
Metrics collection and monitoring for the Enterprise-Ready RAG System.
Supports both local metrics tracking and integration with Prometheus.
"""

import time
import logging
import threading
import json
import os
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Optional imports for external monitoring systems
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Prometheus metrics will not be available.")


class MetricsRegistry:
    """
    Central registry for all metrics in the RAG system.
    Supports both local tracking and Prometheus integration.
    """
    
    def __init__(self, enable_prometheus: bool = False, metrics_dir: str = "data/metrics"):
        """
        Initialize the metrics registry.
        
        Args:
            enable_prometheus: Whether to enable Prometheus metrics
            metrics_dir: Directory to store local metrics data
        """
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Local metrics storage
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)
        self.recent_queries = deque(maxlen=100)  # Store recent queries
        
        # Prometheus metrics (if enabled)
        self.prom_counters = {}
        self.prom_gauges = {}
        self.prom_histograms = {}
        
        # Performance tracking
        self.start_times = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Start Prometheus server if enabled
        if self.enable_prometheus:
            try:
                prometheus_client.start_http_server(8000)
                logger.info("Prometheus metrics server started on port 8000")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {str(e)}")
                self.enable_prometheus = False
        
        # Initialize default metrics
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self):
        """Initialize default metrics for the RAG system."""
        # Document processing metrics
        self.create_counter("documents_processed_total", "Total number of documents processed")
        self.create_counter("documents_indexed_total", "Total number of documents indexed")
        self.create_counter("document_chunks_created_total", "Total number of document chunks created")
        self.create_counter("document_processing_errors_total", "Total number of document processing errors")
        
        # Retrieval metrics
        self.create_counter("queries_total", "Total number of queries processed")
        self.create_counter("retrieval_operations_total", "Total number of retrieval operations")
        self.create_histogram("documents_retrieved_per_query", "Number of documents retrieved per query", 
                              buckets=[1, 3, 5, 10, 20, 50])
        
        # Performance metrics
        self.create_histogram("document_processing_time_seconds", "Time to process a document", 
                              buckets=[0.01, 0.1, 0.5, 1, 5, 10, 30])
        self.create_histogram("retrieval_latency_seconds", "Retrieval latency", 
                              buckets=[0.001, 0.01, 0.1, 0.5, 1, 3])
        self.create_histogram("augmentation_latency_seconds", "Augmentation latency", 
                              buckets=[0.001, 0.01, 0.1, 0.5, 1])
        self.create_histogram("generation_latency_seconds", "Generation latency", 
                              buckets=[0.1, 0.5, 1, 3, 5, 10])
        self.create_histogram("total_query_latency_seconds", "Total query latency", 
                              buckets=[0.1, 0.5, 1, 3, 5, 10, 30])
        
        # Cache metrics
        self.create_gauge("cache_size", "Current size of the cache")
        self.create_counter("cache_hits_total", "Total number of cache hits")
        self.create_counter("cache_misses_total", "Total number of cache misses")
        
        # Quality metrics
        self.create_gauge("retrieval_precision", "Average retrieval precision score")
        self.create_gauge("answer_relevance", "Average answer relevance score")
        self.create_gauge("factual_correctness", "Average factual correctness score")
        self.create_gauge("hallucination_rate", "Average hallucination rate")
        
        # System metrics
        self.create_gauge("vector_store_size_bytes", "Size of the vector store in bytes")
        self.create_gauge("index_count", "Number of indices in the system")
        self.create_gauge("document_count", "Total number of documents in the system")
    
    def create_counter(self, name: str, description: str, labels: List[str] = None) -> None:
        """
        Create a counter metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional list of label names
        """
        with self.lock:
            if self.enable_prometheus:
                self.prom_counters[name] = Counter(name, description, labels or [])
    
    def create_gauge(self, name: str, description: str, labels: List[str] = None) -> None:
        """
        Create a gauge metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional list of label names
        """
        with self.lock:
            if self.enable_prometheus:
                self.prom_gauges[name] = Gauge(name, description, labels or [])
    
    def create_histogram(self, name: str, description: str, buckets: List[float] = None, 
                         labels: List[str] = None) -> None:
        """
        Create a histogram metric.
        
        Args:
            name: Metric name
            description: Metric description
            buckets: Optional list of bucket boundaries
            labels: Optional list of label names
        """
        with self.lock:
            if self.enable_prometheus:
                self.prom_histograms[name] = Histogram(
                    name, description, labels or [], 
                    buckets=buckets or [0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 600]
                )
    
    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Value to increment by
            labels: Optional label values
        """
        with self.lock:
            # Update local metrics
            if labels:
                label_str = json.dumps(labels, sort_keys=True)
                self.counters[f"{name}_{label_str}"] += value
            else:
                self.counters[name] += value
            
            # Update Prometheus metrics
            if self.enable_prometheus and name in self.prom_counters:
                if labels:
                    self.prom_counters[name].labels(**labels).inc(value)
                else:
                    self.prom_counters[name].inc(value)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """
        Set a gauge metric.
        
        Args:
            name: Metric name
            value: Value to set
            labels: Optional label values
        """
        with self.lock:
            # Update local metrics
            if labels:
                label_str = json.dumps(labels, sort_keys=True)
                self.gauges[f"{name}_{label_str}"] = value
            else:
                self.gauges[name] = value
            
            # Update Prometheus metrics
            if self.enable_prometheus and name in self.prom_gauges:
                if labels:
                    self.prom_gauges[name].labels(**labels).set(value)
                else:
                    self.prom_gauges[name].set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """
        Observe a value for a histogram metric.
        
        Args:
            name: Metric name
            value: Value to observe
            labels: Optional label values
        """
        with self.lock:
            # Update local metrics
            if labels:
                label_str = json.dumps(labels, sort_keys=True)
                self.histograms[f"{name}_{label_str}"].append(value)
            else:
                self.histograms[name].append(value)
            
            # Update Prometheus metrics
            if self.enable_prometheus and name in self.prom_histograms:
                if labels:
                    self.prom_histograms[name].labels(**labels).observe(value)
                else:
                    self.prom_histograms[name].observe(value)
    
    def start_timer(self, name: str, labels: Dict[str, str] = None) -> str:
        """
        Start a timer for measuring operation duration.
        
        Args:
            name: Metric name for the timer
            labels: Optional label values
            
        Returns:
            Timer ID
        """
        timer_id = f"{name}_{time.time()}_{threading.get_ident()}"
        
        with self.lock:
            if labels:
                label_str = json.dumps(labels, sort_keys=True)
                self.start_times[f"{timer_id}_{label_str}"] = time.time()
            else:
                self.start_times[timer_id] = time.time()
        
        return timer_id
    
    def stop_timer(self, timer_id: str, labels: Dict[str, str] = None) -> float:
        """
        Stop a timer and record the duration.
        
        Args:
            timer_id: Timer ID returned by start_timer
            labels: Optional label values
            
        Returns:
            Duration in seconds
        """
        with self.lock:
            key = timer_id
            if labels:
                label_str = json.dumps(labels, sort_keys=True)
                key = f"{timer_id}_{label_str}"
            
            if key not in self.start_times:
                logger.warning(f"Timer {key} not found")
                return 0
            
            start_time = self.start_times.pop(key)
            duration = time.time() - start_time
            
            # Extract metric name from timer ID
            metric_name = timer_id.split('_')[0]
            
            # Observe duration in appropriate histogram
            self.observe_histogram(f"{metric_name}_seconds", duration, labels)
            
            return duration
    
    def record_query(self, query: str, response: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """
        Record information about a query for monitoring.
        
        Args:
            query: Query text
            response: Response data
            metrics: Performance metrics
        """
        with self.lock:
            # Record basic query metrics
            self.increment_counter("queries_total")
            
            # Record latency metrics
            if "retrieval_time" in metrics:
                self.observe_histogram("retrieval_latency_seconds", metrics["retrieval_time"])
            
            if "augmentation_time" in metrics:
                self.observe_histogram("augmentation_latency_seconds", metrics["augmentation_time"])
            
            if "generation_time" in metrics:
                self.observe_histogram("generation_latency_seconds", metrics["generation_time"])
            
            if "total_time" in metrics:
                self.observe_histogram("total_query_latency_seconds", metrics["total_time"])
            
            # Record documents retrieved
            if "num_docs_retrieved" in metrics:
                self.observe_histogram("documents_retrieved_per_query", metrics["num_docs_retrieved"])
            
            # Store recent query info
            timestamp = datetime.now().isoformat()
            query_info = {
                "timestamp": timestamp,
                "query": query,
                "metrics": metrics
            }
            
            if "answer" in response:
                query_info["answer_length"] = len(response["answer"])
            
            self.recent_queries.append(query_info)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics values.
        
        Returns:
            Dictionary of all metrics
        """
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: {
                        "count": len(values),
                        "sum": sum(values),
                        "avg": sum(values) / len(values) if values else 0,
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0
                    }
                    for name, values in self.histograms.items()
                }
            }
    
    def save_metrics(self, filename: str = None) -> str:
        """
        Save current metrics to a file.
        
        Args:
            filename: Optional filename, defaults to timestamp-based name
            
        Returns:
            Path to saved metrics file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.metrics_dir / filename
        
        with self.lock:
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": self.get_metrics(),
                "recent_queries": list(self.recent_queries)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2)
        
        return str(filepath)
    
    def reset_metrics(self) -> None:
        """Reset all metrics to their initial values."""
        with self.lock:
            self.counters = defaultdict(int)
            self.gauges = {}
            self.histograms = defaultdict(list)
            self.recent_queries.clear()
            self.start_times = {}


class QueryMonitor:
    """
    Monitor for tracking query performance and metrics.
    """
    
    def __init__(self, metrics_registry: MetricsRegistry):
        """
        Initialize the query monitor.
        
        Args:
            metrics_registry: Metrics registry instance
        """
        self.metrics_registry = metrics_registry
        self.current_query = None
        self.start_time = None
        self.phase_times = {}
    
    def start_query(self, query: str) -> None:
        """
        Start monitoring a new query.
        
        Args:
            query: Query text
        """
        self.current_query = query
        self.start_time = time.time()
        self.phase_times = {}
        
        # Start overall timer
        self.metrics_registry.start_timer("total_query_latency", {"query_type": "rag"})
    
    def start_phase(self, phase: str) -> None:
        """
        Start timing a phase of query processing.
        
        Args:
            phase: Phase name (e.g., "retrieval", "augmentation", "generation")
        """
        if not self.current_query:
            return
        
        timer_id = self.metrics_registry.start_timer(f"{phase}_latency")
        self.phase_times[phase] = {"timer_id": timer_id, "start": time.time()}
    
    def end_phase(self, phase: str, metadata: Dict[str, Any] = None) -> float:
        """
        End timing a phase of query processing.
        
        Args:
            phase: Phase name
            metadata: Optional metadata about the phase
            
        Returns:
            Duration of the phase in seconds
        """
        if not self.current_query or phase not in self.phase_times:
            return 0
        
        phase_data = self.phase_times[phase]
        duration = time.time() - phase_data["start"]
        
        # Stop timer in metrics registry
        self.metrics_registry.stop_timer(phase_data["timer_id"])
        
        # Store metadata
        phase_data["duration"] = duration
        if metadata:
            phase_data["metadata"] = metadata
        
        return duration
    
    def end_query(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        End monitoring the current query.
        
        Args:
            response: Query response
            
        Returns:
            Query metrics
        """
        if not self.current_query:
            return {}
        
        total_time = time.time() - self.start_time
        
        # Collect metrics
        metrics = {
            "total_time": total_time,
            "phases": {}
        }
        
        for phase, data in self.phase_times.items():
            metrics["phases"][phase] = {
                "time": data.get("duration", 0)
            }
            if "metadata" in data:
                metrics["phases"][phase].update(data["metadata"])
            
            # Add to flat metrics dict for easier recording
            metrics[f"{phase}_time"] = data.get("duration", 0)
        
        # Extract additional metrics from response
        if "sources" in response:
            metrics["num_docs_retrieved"] = len(response["sources"])
        
        # Record query in metrics registry
        self.metrics_registry.record_query(self.current_query, response, metrics)
        
        # Reset state
        self.current_query = None
        self.start_time = None
        self.phase_times = {}
        
        return metrics


class SystemMonitor:
    """
    Monitor for tracking system-level metrics.
    """
    
    def __init__(self, metrics_registry: MetricsRegistry, collection_interval: int = 60):
        """
        Initialize the system monitor.
        
        Args:
            metrics_registry: Metrics registry instance
            collection_interval: Interval in seconds for collecting system metrics
        """
        self.metrics_registry = metrics_registry
        self.collection_interval = collection_interval
        self.running = False
        self.thread = None
        self.hooks = []
        
        # Initialize system metrics
        self._initialize_system_metrics()
    
    def _initialize_system_metrics(self):
        """Initialize system-level metrics."""
        # Resource metrics
        self.metrics_registry.create_gauge("cpu_usage_percent", "CPU usage percentage")
        self.metrics_registry.create_gauge("memory_usage_bytes", "Memory usage in bytes")
        self.metrics_registry.create_gauge("disk_usage_bytes", "Disk usage in bytes")
        
        # Process metrics
        self.metrics_registry.create_gauge("process_cpu_percent", "Process CPU usage percentage")
        self.metrics_registry.create_gauge("process_memory_bytes", "Process memory usage in bytes")
        self.metrics_registry.create_gauge("process_threads", "Number of threads in the process")
        
        # System load
        self.metrics_registry.create_gauge("system_load_1min", "System load average (1 min)")
        self.metrics_registry.create_gauge("system_load_5min", "System load average (5 min)")
        self.metrics_registry.create_gauge("system_load_15min", "System load average (15 min)")
    
    def add_collection_hook(self, hook: Callable[[], Dict[str, float]]) -> None:
        """
        Add a hook for collecting additional metrics.
        
        Args:
            hook: Function that returns a dictionary of metric name -> value
        """
        self.hooks.append(hook)
    
    def collect_system_metrics(self) -> None:
        """Collect system metrics."""
        try:
            # Try to import psutil for system metrics
            import psutil
            
            # CPU metrics
            self.metrics_registry.set_gauge("cpu_usage_percent", psutil.cpu_percent())
            
            # Memory metrics
            mem = psutil.virtual_memory()
            self.metrics_registry.set_gauge("memory_usage_bytes", mem.used)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics_registry.set_gauge("disk_usage_bytes", disk.used)
            
            # Process metrics
            process = psutil.Process()
            self.metrics_registry.set_gauge("process_cpu_percent", process.cpu_percent())
            self.metrics_registry.set_gauge("process_memory_bytes", process.memory_info().rss)
            self.metrics_registry.set_gauge("process_threads", process.num_threads())
            
            # System load
            load1, load5, load15 = psutil.getloadavg()
            self.metrics_registry.set_gauge("system_load_1min", load1)
            self.metrics_registry.set_gauge("system_load_5min", load5)
            self.metrics_registry.set_gauge("system_load_15min", load15)
        
        except ImportError:
            logger.warning("psutil not installed. System metrics will not be collected.")
        
        # Call collection hooks
        for hook in self.hooks:
            try:
                metrics = hook()
                for name, value in metrics.items():
                    self.metrics_registry.set_gauge(name, value)
            except Exception as e:
                logger.error(f"Error in metrics collection hook: {str(e)}")
    
    def _collection_loop(self) -> None:
        """Background loop for collecting metrics."""
        while self.running:
            try:
                self.collect_system_metrics()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {str(e)}")
            
            # Sleep until next collection interval
            for _ in range(self.collection_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def start(self) -> None:
        """Start the system monitor."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"System monitor started with collection interval {self.collection_interval}s")
    
    def stop(self) -> None:
        """Stop the system monitor."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        
        logger.info("System monitor stopped")


# Global metrics registry instance
_metrics_registry = None

def get_metrics_registry(enable_prometheus: bool = False, metrics_dir: str = "data/metrics") -> MetricsRegistry:
    """
    Get the global metrics registry instance.
    
    Args:
        enable_prometheus: Whether to enable Prometheus metrics
        metrics_dir: Directory to store local metrics data
        
    Returns:
        Metrics registry instance
    """
    global _metrics_registry
    
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry(
            enable_prometheus=enable_prometheus,
            metrics_dir=metrics_dir
        )
    
    return _metrics_registry
