#!/usr/bin/env python
"""
Web-based dashboard for visualizing Enterprise-Ready RAG System metrics.
Uses Streamlit for a clean, interactive dashboard experience.
"""

import os
import sys
import json
import time
import argparse
import logging
import threading
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit or Plotly not installed. Run 'pip install streamlit plotly' to enable the dashboard.")

# Import local modules
from src.monitoring.metrics import get_metrics_registry, MetricsRegistry


class MetricsDashboard:
    """
    Dashboard for visualizing RAG system metrics.
    """
    
    def __init__(self, metrics_registry: MetricsRegistry = None, metrics_dir: str = "data/metrics"):
        """
        Initialize the metrics dashboard.
        
        Args:
            metrics_registry: Optional metrics registry instance
            metrics_dir: Directory containing metrics data files
        """
        self.metrics_registry = metrics_registry or get_metrics_registry(metrics_dir=metrics_dir)
        self.metrics_dir = Path(metrics_dir)
        
        # Cache for performance metrics
        self.performance_data = []
        self.retrieval_data = []
        self.query_history = []
        self.last_update = datetime.now()
        
        # Create data directory if it doesn't exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def load_metrics_files(self, max_files: int = 10) -> List[Dict[str, Any]]:
        """
        Load metrics data from files.
        
        Args:
            max_files: Maximum number of files to load
            
        Returns:
            List of metrics data dictionaries
        """
        metrics_data = []
        
        try:
            # Get list of metrics files sorted by modification time (newest first)
            files = sorted(
                [f for f in self.metrics_dir.glob("metrics_*.json")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Load data from each file
            for file_path in files[:max_files]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data["file"] = str(file_path)
                        metrics_data.append(data)
                except Exception as e:
                    logger.error(f"Error loading metrics file {file_path}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error listing metrics files: {str(e)}")
        
        return metrics_data
    
    def get_performance_data(self) -> pd.DataFrame:
        """
        Get performance metrics data.
        
        Returns:
            DataFrame with performance metrics
        """
        # Get current metrics
        current_metrics = self.metrics_registry.get_metrics()
        
        # Extract histogram data
        latency_metrics = {}
        for name, values in current_metrics.get("histograms", {}).items():
            if "_latency_seconds" in name or "_time_seconds" in name:
                latency_metrics[name] = values
        
        # Convert to DataFrame
        data = []
        timestamp = datetime.now().isoformat()
        
        for metric_name, metric_data in latency_metrics.items():
            if "avg" in metric_data:
                # Convert to milliseconds for better readability
                value_ms = metric_data["avg"] * 1000
                data.append({
                    "timestamp": timestamp,
                    "metric": metric_name.replace("_seconds", "").replace("_latency", ""),
                    "value_ms": value_ms,
                    "count": metric_data.get("count", 0)
                })
        
        # Add data to performance history
        self.performance_data.extend(data)
        
        # Keep only the last 1000 data points
        if len(self.performance_data) > 1000:
            self.performance_data = self.performance_data[-1000:]
        
        return pd.DataFrame(self.performance_data)
    
    def get_retrieval_data(self) -> pd.DataFrame:
        """
        Get retrieval metrics data.
        
        Returns:
            DataFrame with retrieval metrics
        """
        # Get current metrics
        current_metrics = self.metrics_registry.get_metrics()
        
        # Extract retrieval metrics
        retrieval_gauges = {}
        for name, value in current_metrics.get("gauges", {}).items():
            if "retrieval" in name or "precision" in name or "relevance" in name:
                retrieval_gauges[name] = value
        
        # Convert to DataFrame
        data = []
        timestamp = datetime.now().isoformat()
        
        for metric_name, value in retrieval_gauges.items():
            data.append({
                "timestamp": timestamp,
                "metric": metric_name,
                "value": value
            })
        
        # Add data to retrieval history
        self.retrieval_data.extend(data)
        
        # Keep only the last 1000 data points
        if len(self.retrieval_data) > 1000:
            self.retrieval_data = self.retrieval_data[-1000:]
        
        return pd.DataFrame(self.retrieval_data)
    
    def get_query_history(self) -> pd.DataFrame:
        """
        Get query history data.
        
        Returns:
            DataFrame with query history
        """
        # Get recent queries from the metrics registry
        recent_queries = list(getattr(self.metrics_registry, "recent_queries", []))
        
        # Convert to DataFrame
        data = []
        
        for query_info in recent_queries:
            entry = {
                "timestamp": query_info.get("timestamp", ""),
                "query": query_info.get("query", ""),
                "total_time_ms": query_info.get("metrics", {}).get("total_time", 0) * 1000,
                "retrieval_time_ms": query_info.get("metrics", {}).get("retrieval_time", 0) * 1000,
                "generation_time_ms": query_info.get("metrics", {}).get("generation_time", 0) * 1000,
                "num_docs_retrieved": query_info.get("metrics", {}).get("num_docs_retrieved", 0),
                "answer_length": query_info.get("answer_length", 0)
            }
            data.append(entry)
        
        # Update query history
        self.query_history = data
        
        return pd.DataFrame(data)
    
    def update_data(self) -> None:
        """Update all data from the metrics registry."""
        self.get_performance_data()
        self.get_retrieval_data()
        self.get_query_history()
        self.last_update = datetime.now()
    
    def render_streamlit_dashboard(self) -> None:
        """Render the Streamlit dashboard."""
        if not STREAMLIT_AVAILABLE:
            logger.error("Streamlit is not available. Install with 'pip install streamlit plotly'")
            return
        
        # Update data if it's been more than 5 seconds since the last update
        if (datetime.now() - self.last_update).total_seconds() > 5:
            self.update_data()
        
        # Configure page
        st.set_page_config(
            page_title="RAG System Metrics Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        # Title and description
        st.title("📊 Enterprise-Ready RAG System Dashboard")
        st.markdown(
            """
            Real-time monitoring and metrics for the Enterprise-Ready RAG System.
            """
        )
        
        # Sidebar
        with st.sidebar:
            st.header("Controls")
            
            # Refresh button
            if st.button("Refresh Data"):
                self.update_data()
            
            st.markdown("---")
            
            # Time range selector
            time_range = st.selectbox(
                "Time Range", 
                ["Last 15 minutes", "Last hour", "Last 4 hours", "Last 24 hours", "All time"],
                index=0
            )
            
            # Auto-refresh
            auto_refresh = st.checkbox("Auto-refresh (10s)", value=True)
            if auto_refresh:
                time.sleep(0.1)  # Small delay
                st.rerun()
            
            st.markdown("---")
            
            # Metrics file selection
            st.header("Historical Data")
            metrics_files = self.load_metrics_files()
            if metrics_files:
                file_options = [f["file"] for f in metrics_files]
                selected_file = st.selectbox("Load metrics file", file_options)
                
                if st.button("Load Selected File"):
                    # Load data from the selected file
                    for file_data in metrics_files:
                        if file_data["file"] == selected_file:
                            # Display file info
                            st.info(f"Loaded data from {selected_file}")
                            st.write(f"Timestamp: {file_data.get('timestamp', 'Unknown')}")
            else:
                st.info("No metrics files found")
        
        # Apply time range filter
        def filter_by_time_range(df, timestamp_col="timestamp"):
            if df.empty or timestamp_col not in df.columns:
                return df
            
            # Convert timestamp column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            now = pd.Timestamp.now()
            
            if time_range == "Last 15 minutes":
                return df[df[timestamp_col] > now - pd.Timedelta(minutes=15)]
            elif time_range == "Last hour":
                return df[df[timestamp_col] > now - pd.Timedelta(hours=1)]
            elif time_range == "Last 4 hours":
                return df[df[timestamp_col] > now - pd.Timedelta(hours=4)]
            elif time_range == "Last 24 hours":
                return df[df[timestamp_col] > now - pd.Timedelta(days=1)]
            else:
                return df
        
        # Main content
        col1, col2 = st.columns(2)
        
        # System overview
        with col1:
            st.header("System Overview")
            
            # Get current metrics
            current_metrics = self.metrics_registry.get_metrics()
            
            # Create metrics cards
            metric_cols = st.columns(3)
            
            # Query count
            with metric_cols[0]:
                query_count = current_metrics.get("counters", {}).get("queries_total", 0)
                st.metric("Total Queries", query_count)
            
            # Documents processed
            with metric_cols[1]:
                docs_processed = current_metrics.get("counters", {}).get("documents_processed_total", 0)
                st.metric("Documents Processed", docs_processed)
            
            # Documents indexed
            with metric_cols[2]:
                docs_indexed = current_metrics.get("counters", {}).get("documents_indexed_total", 0)
                st.metric("Documents Indexed", docs_indexed)
            
            # Cache metrics
            cache_cols = st.columns(3)
            
            # Cache hits
            with cache_cols[0]:
                cache_hits = current_metrics.get("counters", {}).get("cache_hits_total", 0)
                st.metric("Cache Hits", cache_hits)
            
            # Cache misses
            with cache_cols[1]:
                cache_misses = current_metrics.get("counters", {}).get("cache_misses_total", 0)
                st.metric("Cache Misses", cache_misses)
            
            # Cache hit ratio
            with cache_cols[2]:
                if cache_hits + cache_misses > 0:
                    hit_ratio = cache_hits / (cache_hits + cache_misses) * 100
                    st.metric("Cache Hit Ratio", f"{hit_ratio:.1f}%")
                else:
                    st.metric("Cache Hit Ratio", "N/A")
        
        # Quality metrics
        with col2:
            st.header("Quality Metrics")
            
            # Create gauge charts for quality metrics
            quality_cols = st.columns(2)
            
            gauges = current_metrics.get("gauges", {})
            
            # Retrieval precision
            with quality_cols[0]:
                retrieval_precision = gauges.get("retrieval_precision", 0)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=retrieval_precision,
                    title={"text": "Retrieval Precision"},
                    gauge={
                        "axis": {"range": [0, 1], "tickwidth": 1},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 0.6], "color": "red"},
                            {"range": [0.6, 0.8], "color": "orange"},
                            {"range": [0.8, 1], "color": "green"}
                        ]
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Answer relevance
            with quality_cols[1]:
                answer_relevance = gauges.get("answer_relevance", 0)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=answer_relevance,
                    title={"text": "Answer Relevance"},
                    gauge={
                        "axis": {"range": [0, 1], "tickwidth": 1},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 0.6], "color": "red"},
                            {"range": [0.6, 0.8], "color": "orange"},
                            {"range": [0.8, 1], "color": "green"}
                        ]
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # More quality metrics
            quality_cols = st.columns(2)
            
            # Factual correctness
            with quality_cols[0]:
                factual_correctness = gauges.get("factual_correctness", 0)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=factual_correctness,
                    title={"text": "Factual Correctness"},
                    gauge={
                        "axis": {"range": [0, 1], "tickwidth": 1},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 0.6], "color": "red"},
                            {"range": [0.6, 0.8], "color": "orange"},
                            {"range": [0.8, 1], "color": "green"}
                        ]
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Hallucination rate
            with quality_cols[1]:
                hallucination_rate = gauges.get("hallucination_rate", 0)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=hallucination_rate,
                    title={"text": "Hallucination Rate"},
                    gauge={
                        "axis": {"range": [0, 1], "tickwidth": 1},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 0.3], "color": "green"},
                            {"range": [0.3, 0.6], "color": "orange"},
                            {"range": [0.6, 1], "color": "red"}
                        ]
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.header("Performance Metrics")
        
        # Get performance data
        perf_df = pd.DataFrame(self.performance_data)
        if not perf_df.empty and "timestamp" in perf_df.columns:
            # Convert timestamp to datetime
            perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"])
            
            # Filter by time range
            perf_df = filter_by_time_range(perf_df)
            
            # Latency over time chart
            fig = px.line(
                perf_df, 
                x="timestamp", 
                y="value_ms", 
                color="metric",
                title="Latency Over Time",
                labels={"value_ms": "Latency (ms)", "timestamp": "Time", "metric": "Metric"}
            )
            fig.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
            
            # Latency distribution
            st.subheader("Latency Distribution")
            
            # Group by metric and calculate statistics
            latency_stats = perf_df.groupby("metric")["value_ms"].agg(["mean", "min", "max", "std", "count"]).reset_index()
            latency_stats = latency_stats.sort_values("mean", ascending=False)
            
            # Create a bar chart for mean latency
            fig = px.bar(
                latency_stats,
                x="metric",
                y="mean",
                error_y="std",
                title="Average Latency by Component",
                labels={"mean": "Average Latency (ms)", "metric": "Component"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display latency stats table
            latency_stats = latency_stats.rename(columns={
                "mean": "Mean (ms)",
                "min": "Min (ms)",
                "max": "Max (ms)",
                "std": "Std Dev (ms)",
                "count": "Count"
            })
            st.dataframe(latency_stats, use_container_width=True)
        else:
            st.info("No performance data available yet")
        
        # Recent queries
        st.header("Recent Queries")
        
        # Get query history
        query_df = pd.DataFrame(self.query_history)
        if not query_df.empty and "timestamp" in query_df.columns:
            # Convert timestamp to datetime
            query_df["timestamp"] = pd.to_datetime(query_df["timestamp"])
            
            # Filter by time range
            query_df = filter_by_time_range(query_df)
            
            # Sort by timestamp (most recent first)
            query_df = query_df.sort_values("timestamp", ascending=False)
            
            # Display recent queries table
            st.dataframe(
                query_df[["timestamp", "query", "total_time_ms", "retrieval_time_ms", "generation_time_ms", "num_docs_retrieved"]],
                use_container_width=True
            )
            
            # Queries over time
            if len(query_df) > 1:
                # Group by timestamp (hourly) and count queries
                query_df["hour"] = query_df["timestamp"].dt.floor("H")
                query_count_by_hour = query_df.groupby("hour").size().reset_index(name="count")
                
                # Create a bar chart for query count over time
                fig = px.bar(
                    query_count_by_hour,
                    x="hour",
                    y="count",
                    title="Queries Over Time",
                    labels={"count": "Query Count", "hour": "Time"}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No query history available yet")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start the Enterprise-Ready RAG System metrics dashboard"
    )
    
    parser.add_argument(
        "--metrics-dir", type=str, default="data/metrics",
        help="Directory containing metrics data files"
    )
    parser.add_argument(
        "--port", type=int, default=8501,
        help="Port to run the Streamlit dashboard on"
    )
    parser.add_argument(
        "--enable-prometheus", action="store_true",
        help="Enable Prometheus metrics export"
    )
    
    return parser.parse_args()


def run_streamlit(script_path: str, port: int = 8501) -> None:
    """
    Run the Streamlit dashboard.
    
    Args:
        script_path: Path to this script
        port: Port to run on
    """
    import subprocess
    
    try:
        cmd = [
            "streamlit", "run",
            script_path,
            "--server.port", str(port),
            "--browser.serverAddress", "localhost",
            "--server.headless", "true"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info(f"Started Streamlit dashboard on http://localhost:{port}")
        
        # Print URL
        print(f"\n✨ RAG System Dashboard is running at: http://localhost:{port}\n")
        
        # Wait for process to finish
        process.wait()
    
    except Exception as e:
        logger.error(f"Error starting Streamlit: {str(e)}")
        print(f"Error: {str(e)}")
        print("Make sure Streamlit is installed: pip install streamlit plotly")


def main():
    """Main entry point."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is required to run the dashboard.")
        print("Install with: pip install streamlit plotly")
        return
    
    args = parse_args()
    
    # Initialize metrics registry
    metrics_registry = get_metrics_registry(
        enable_prometheus=args.enable_prometheus,
        metrics_dir=args.metrics_dir
    )
    
    # Create dashboard instance
    dashboard = MetricsDashboard(
        metrics_registry=metrics_registry,
        metrics_dir=args.metrics_dir
    )
    
    # Set up Streamlit script to run the dashboard
    if __name__ == "__main__":
        dashboard.render_streamlit_dashboard()
    else:
        # Run in a separate process
        run_streamlit(__file__, port=args.port)


if __name__ == "__main__":
    main()
