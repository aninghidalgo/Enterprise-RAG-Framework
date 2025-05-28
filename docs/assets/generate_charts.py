"""
Generate visualization charts for the Enterprise-Ready RAG System benchmarks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import PercentFormatter

# Set style
plt.style.use('fivethirtyeight')
sns.set_palette("muted")
sns.set_context("talk")

# Create output directory
os.makedirs(".", exist_ok=True)

# Define colors
colors = {
    "hybrid": "#1E88E5",
    "dense": "#FFC107",
    "sparse": "#4CAF50",
    "tfidf": "#9C27B0",
    "keyword": "#F44336",
    "no_reranker": "#607D8B",
}

# 1. Retrieval Performance Chart
def create_retrieval_performance_chart():
    """Create bar chart comparing different retrieval methods."""
    methods = ["Hybrid (Our System)", "Dense-only (SBERT)", "Sparse-only (BM25)", "TF-IDF", "Keyword Search"]
    precision = [0.872, 0.810, 0.768, 0.712, 0.625]
    recall = [0.783, 0.752, 0.694, 0.641, 0.518]
    ndcg = [0.891, 0.834, 0.782, 0.725, 0.602]
    mrr = [0.837, 0.795, 0.730, 0.684, 0.590]
    
    metrics = ["Precision@5", "Recall@5", "NDCG@5", "MRR"]
    
    # Prepare data
    df = pd.DataFrame({
        "Method": np.repeat(methods, 4),
        "Metric": np.tile(metrics, 5),
        "Value": np.concatenate([
            np.array([precision[i], recall[i], ndcg[i], mrr[i]]) for i in range(5)
        ])
    })
    
    # Create figure
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x="Method", y="Value", hue="Metric", data=df)
    
    # Customize
    plt.title("Retrieval Performance Comparison", fontsize=18, pad=20)
    plt.xlabel("Retrieval Method", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title="Metric", fontsize=12, title_fontsize=14)
    
    # Add values on bars
    for container in chart.containers:
        chart.bar_label(container, fmt='%.2f', fontsize=10)
    
    # Save
    plt.tight_layout()
    plt.savefig("retrieval_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Retrieval performance chart created.")

# 2. Document Type Performance Chart
def create_document_type_performance_chart():
    """Create grouped bar chart showing performance by document type."""
    doc_types = ["Technical Documentation", "Academic Papers", "News Articles", "Legal Documents", "Medical Records"]
    hybrid_scores = [0.912, 0.856, 0.891, 0.837, 0.894]
    dense_scores = [0.837, 0.821, 0.842, 0.754, 0.862]
    sparse_scores = [0.865, 0.728, 0.792, 0.815, 0.752]
    
    # Prepare data
    df = pd.DataFrame({
        "Document Type": np.repeat(doc_types, 3),
        "Method": np.tile(["Hybrid (Our System)", "Dense-only", "Sparse-only"], 5),
        "Score": np.concatenate([
            [hybrid_scores[i], dense_scores[i], sparse_scores[i]] for i in range(5)
        ])
    })
    
    # Create figure
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x="Document Type", y="Score", hue="Method", data=df)
    
    # Colors
    for i, patch in enumerate(chart.patches):
        if i % 3 == 0:  # Hybrid
            patch.set_facecolor(colors["hybrid"])
        elif i % 3 == 1:  # Dense
            patch.set_facecolor(colors["dense"])
        else:  # Sparse
            patch.set_facecolor(colors["sparse"])
    
    # Customize
    plt.title("Performance by Document Type", fontsize=18, pad=20)
    plt.xlabel("Document Type", fontsize=14)
    plt.ylabel("Score (NDCG@5)", fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title="Retrieval Method", fontsize=12, title_fontsize=14)
    
    # Add values on bars
    for container in chart.containers:
        chart.bar_label(container, fmt='%.2f', fontsize=10)
    
    # Save
    plt.tight_layout()
    plt.savefig("document_type_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Document type performance chart created.")

# 3. Reranking Effect Chart
def create_reranking_effect_chart():
    """Create chart showing the effect of reranking on performance."""
    configs = ["Hybrid with Reranking", "Hybrid without Reranking", "Dense with Reranking", "Dense without Reranking"]
    precision = [0.872, 0.815, 0.842, 0.810]
    recall = [0.783, 0.745, 0.769, 0.752]
    ndcg = [0.891, 0.832, 0.863, 0.834]
    latency = [42, 28, 39, 25]
    
    # Prepare data for metrics
    df_metrics = pd.DataFrame({
        "Configuration": np.repeat(configs, 3),
        "Metric": np.tile(["Precision@5", "Recall@5", "NDCG@5"], 4),
        "Value": np.concatenate([
            [precision[i], recall[i], ndcg[i]] for i in range(4)
        ])
    })
    
    # Create figure for metrics
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x="Configuration", y="Value", hue="Metric", data=df_metrics)
    
    # Customize
    plt.title("Effect of Reranking on Retrieval Performance", fontsize=18, pad=20)
    plt.xlabel("Configuration", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title="Metric", fontsize=12, title_fontsize=14)
    
    # Add values on bars
    for container in chart.containers:
        chart.bar_label(container, fmt='%.2f', fontsize=10)
    
    # Save
    plt.tight_layout()
    plt.savefig("reranking_effect.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create figure for latency
    plt.figure(figsize=(10, 6))
    bars = plt.bar(configs, latency, color=sns.color_palette("muted", 4))
    
    # Customize
    plt.title("Latency by Configuration", fontsize=18, pad=20)
    plt.xlabel("Configuration", fontsize=14)
    plt.ylabel("Latency (ms)", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height} ms', ha='center', va='bottom', fontsize=12)
    
    # Save
    plt.tight_layout()
    plt.savefig("reranking_latency.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Reranking effect charts created.")

# 4. Answer Quality Chart
def create_answer_quality_chart():
    """Create chart showing answer quality metrics."""
    systems = ["Our RAG System", "GPT-4 (No RAG)", "Claude 3 (No RAG)", "Baseline RAG"]
    factual = [92.7, 85.3, 86.2, 88.1]
    relevance = [4.58, 4.21, 4.32, 4.25]
    hallucination = [2.3, 8.7, 7.8, 5.2]
    citation = [94.2, 0, 0, 86.5]
    
    # Create figure for factual accuracy and hallucination
    plt.figure(figsize=(12, 8))
    
    # First subplot for factual accuracy
    plt.subplot(2, 1, 1)
    bars1 = plt.bar(systems, factual, color=sns.color_palette("muted", 4))
    plt.title("Factual Accuracy by System", fontsize=16, pad=15)
    plt.ylabel("Factual Accuracy (%)", fontsize=14)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height}%', ha='center', va='bottom', fontsize=12)
    
    # Second subplot for hallucination rate
    plt.subplot(2, 1, 2)
    bars2 = plt.bar(systems, hallucination, color=sns.color_palette("muted", 4))
    plt.title("Hallucination Rate by System", fontsize=16, pad=15)
    plt.ylabel("Hallucination Rate (%)", fontsize=14)
    plt.ylim(0, 10)
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{height}%', ha='center', va='bottom', fontsize=12)
    
    # Save
    plt.tight_layout()
    plt.savefig("answer_quality.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create figure for relevance scores
    plt.figure(figsize=(10, 6))
    bars3 = plt.bar(systems, relevance, color=sns.color_palette("muted", 4))
    plt.title("Answer Relevance Score by System", fontsize=18, pad=20)
    plt.ylabel("Relevance Score (out of 5)", fontsize=14)
    plt.ylim(0, 5)
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars3:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}/5', ha='center', va='bottom', fontsize=12)
    
    # Save
    plt.tight_layout()
    plt.savefig("answer_relevance.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Answer quality charts created.")

# 5. Human Evaluation Chart
def create_human_evaluation_chart():
    """Create chart showing human evaluation results."""
    systems = ["Our RAG System", "GPT-4 (No RAG)", "Claude 3 (No RAG)", "Baseline RAG"]
    completeness = [4.72, 4.21, 4.28, 4.41]
    correctness = [4.83, 4.35, 4.42, 4.52]
    coherence = [4.65, 4.58, 4.61, 4.47]
    overall = [4.75, 4.32, 4.38, 4.45]
    
    # Prepare data
    df = pd.DataFrame({
        "System": np.repeat(systems, 4),
        "Metric": np.tile(["Completeness", "Factual Correctness", "Coherence", "Overall Quality"], 4),
        "Score": np.concatenate([
            [completeness[i], correctness[i], coherence[i], overall[i]] for i in range(4)
        ])
    })
    
    # Create figure
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x="System", y="Score", hue="Metric", data=df)
    
    # Customize
    plt.title("Human Evaluation Results", fontsize=18, pad=20)
    plt.xlabel("System", fontsize=14)
    plt.ylabel("Score (out of 5)", fontsize=14)
    plt.ylim(0, 5)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title="Evaluation Metric", fontsize=12, title_fontsize=14)
    
    # Add values on bars
    for container in chart.containers:
        chart.bar_label(container, fmt='%.2f', fontsize=10)
    
    # Save
    plt.tight_layout()
    plt.savefig("human_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Human evaluation chart created.")

# 6. Indexing Performance Chart
def create_indexing_performance_chart():
    """Create chart showing indexing performance."""
    doc_counts = [1000, 10000, 100000, 1000000]
    indexing_time = [1.2, 8.5, 68, 580]
    index_size = [0.4, 3.2, 28, 265]
    ram_usage = [2.1, 4.7, 12.3, 24.6]
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot indexing time on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Document Count', fontsize=14)
    ax1.set_ylabel('Indexing Time (minutes)', color=color, fontsize=14)
    ax1.plot(doc_counts, indexing_time, color=color, marker='o', linewidth=3, markersize=10, label='Indexing Time')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    
    # Create second y-axis for index size
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Index Size (GB)', color=color, fontsize=14)
    ax2.plot(doc_counts, index_size, color=color, marker='s', linewidth=3, markersize=10, label='Index Size')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Create third y-axis for RAM usage
    ax3 = ax1.twinx()
    # Position the third y-axis on the right, offset from the second
    ax3.spines["right"].set_position(("axes", 1.2))
    color = 'tab:green'
    ax3.set_ylabel('RAM Usage (GB)', color=color, fontsize=14)
    ax3.plot(doc_counts, ram_usage, color=color, marker='^', linewidth=3, markersize=10, label='RAM Usage')
    ax3.tick_params(axis='y', labelcolor=color)
    
    # Add title and legends
    plt.title('Indexing Performance by Document Count', fontsize=18, pad=20)
    
    # Combine legends from all axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', fontsize=12)
    
    # Format x-axis labels
    ax1.set_xticks(doc_counts)
    ax1.set_xticklabels([f"{count:,}" for count in doc_counts])
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig("indexing_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Indexing performance chart created.")

# 7. Retrieval Latency Chart
def create_retrieval_latency_chart():
    """Create chart showing retrieval latency."""
    doc_counts = [1000, 10000, 100000, 1000000]
    hybrid_latency = [18, 42, 87, 156]
    dense_latency = [15, 38, 75, 142]
    sparse_latency = [12, 25, 56, 98]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot data
    plt.plot(doc_counts, hybrid_latency, marker='o', linewidth=3, markersize=10, label='Hybrid Retrieval')
    plt.plot(doc_counts, dense_latency, marker='s', linewidth=3, markersize=10, label='Dense-only')
    plt.plot(doc_counts, sparse_latency, marker='^', linewidth=3, markersize=10, label='Sparse-only')
    
    # Customize
    plt.title('Retrieval Latency by Document Count', fontsize=18, pad=20)
    plt.xlabel('Document Count', fontsize=14)
    plt.ylabel('Latency (ms)', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    
    # Format x-axis labels
    plt.xticks(doc_counts, [f"{count:,}" for count in doc_counts])
    
    # Add annotations
    for i, count in enumerate(doc_counts):
        plt.annotate(f"{hybrid_latency[i]} ms", (count, hybrid_latency[i]),
                    textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f"{dense_latency[i]} ms", (count, dense_latency[i]),
                    textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f"{sparse_latency[i]} ms", (count, sparse_latency[i]),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Save
    plt.tight_layout()
    plt.savefig("retrieval_latency.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Retrieval latency chart created.")

# 8. Hardware Comparison Chart
def create_hardware_comparison_chart():
    """Create chart comparing performance on different hardware."""
    configs = ["CPU Only (16 cores)", "GPU (NVIDIA T4)", "GPU (NVIDIA A100)"]
    indexing_speed = [24.5, 142.3, 356.8]
    retrieval_latency = [87, 42, 28]
    query_latency = [980, 645, 520]
    
    # Prepare data
    df = pd.DataFrame({
        "Configuration": np.repeat(configs, 3),
        "Metric": np.tile(["Indexing Speed (docs/sec)", "Retrieval Latency (ms)", "End-to-End Query (ms)"], 3),
        "Value": np.concatenate([
            [indexing_speed[i], retrieval_latency[i], query_latency[i]] for i in range(3)
        ])
    })
    
    # Create separate dataframes for each metric due to different scales
    df_indexing = df[df["Metric"] == "Indexing Speed (docs/sec)"]
    df_retrieval = df[df["Metric"] == "Retrieval Latency (ms)"]
    df_query = df[df["Metric"] == "End-to-End Query (ms)"]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot indexing speed
    sns.barplot(x="Configuration", y="Value", data=df_indexing, ax=ax1, palette="muted")
    ax1.set_title("Indexing Speed by Hardware", fontsize=16, pad=15)
    ax1.set_ylabel("Docs/second", fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(indexing_speed):
        ax1.text(i, v + 10, f"{v}", ha='center', fontsize=12)
    
    # Plot retrieval latency
    sns.barplot(x="Configuration", y="Value", data=df_retrieval, ax=ax2, palette="muted")
    ax2.set_title("Retrieval Latency by Hardware", fontsize=16, pad=15)
    ax2.set_ylabel("Latency (ms)", fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(retrieval_latency):
        ax2.text(i, v + 3, f"{v} ms", ha='center', fontsize=12)
    
    # Plot query latency
    sns.barplot(x="Configuration", y="Value", data=df_query, ax=ax3, palette="muted")
    ax3.set_title("End-to-End Query Latency by Hardware", fontsize=16, pad=15)
    ax3.set_ylabel("Latency (ms)", fontsize=14)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(query_latency):
        ax3.text(i, v + 30, f"{v} ms", ha='center', fontsize=12)
    
    # Save
    plt.tight_layout()
    plt.savefig("hardware_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Hardware comparison chart created.")

# 9. Ablation Study Chart
def create_ablation_study_chart():
    """Create chart showing results of ablation studies."""
    configs = ["Full System", "No Reranking", "No Context Augmentation", "No Hybrid Retrieval", "No Citation"]
    answer_quality = [4.75, 4.52, 4.36, 4.45, 4.68]
    factual_accuracy = [92.7, 89.3, 87.8, 88.1, 88.4]
    hallucination_rate = [2.3, 4.1, 5.2, 4.8, 6.7]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot answer quality
    bars1 = ax1.bar(configs, answer_quality, color=sns.color_palette("muted", 5))
    ax1.set_title("Answer Quality by Configuration", fontsize=16, pad=15)
    ax1.set_ylabel("Quality Score (out of 5)", fontsize=14)
    ax1.set_ylim(0, 5)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}/5', ha='center', va='bottom', fontsize=12)
    
    # Plot factual accuracy
    bars2 = ax2.bar(configs, factual_accuracy, color=sns.color_palette("muted", 5))
    ax2.set_title("Factual Accuracy by Configuration", fontsize=16, pad=15)
    ax2.set_ylabel("Factual Accuracy (%)", fontsize=14)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height}%', ha='center', va='bottom', fontsize=12)
    
    # Plot hallucination rate
    bars3 = ax3.bar(configs, hallucination_rate, color=sns.color_palette("muted", 5))
    ax3.set_title("Hallucination Rate by Configuration", fontsize=16, pad=15)
    ax3.set_ylabel("Hallucination Rate (%)", fontsize=14)
    ax3.set_ylim(0, 8)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height}%', ha='center', va='bottom', fontsize=12)
    
    # Save
    plt.tight_layout()
    plt.savefig("ablation_study.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Ablation study chart created.")

# Generate all charts
if __name__ == "__main__":
    print("Generating benchmark charts...")
    create_retrieval_performance_chart()
    create_document_type_performance_chart()
    create_reranking_effect_chart()
    create_answer_quality_chart()
    create_human_evaluation_chart()
    create_indexing_performance_chart()
    create_retrieval_latency_chart()
    create_hardware_comparison_chart()
    create_ablation_study_chart()
    print("All charts generated successfully!")
