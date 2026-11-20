"""
Module trực quan hóa dữ liệu và kết quả thực nghiệm:
- Biểu đồ EDA (Top 20 mặt hàng phổ biến, phân phối kích thước giỏ hàng)
- Biểu đồ so sánh Runtime (ms) & Peak Memory (MB) giữa Apriori và FP-Growth
- Heatmap số lượng luật theo lưới Support - Confidence
- Biểu đồ Scatter Plot & Network Graph cho các luật kết hợp có Lift > 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path

# Thiết lập giao diện biểu đồ chuyên nghiệp
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10
sns.set_theme(style="whitegrid")


def plot_top_items(df: pd.DataFrame, output_dir: Path, top_n: int = 20) -> Path:
    """Vẽ biểu đồ Top N mặt hàng xuất hiện nhiều nhất và phân phối độ dài giỏ hàng."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Top N mặt hàng phổ biến nhất
    item_counts = df["itemDescription"].value_counts().head(top_n)
    sns.barplot(
        x=item_counts.values,
        y=item_counts.index,
        hue=item_counts.index,
        palette="viridis",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title(f"Top {top_n} Most Frequent Items in Groceries Dataset", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Purchase Count")
    axes[0].set_ylabel("Item Description")

    # 2. Phân phối kích thước giỏ hàng (Số mặt hàng / giao dịch)
    basket_sizes = df.groupby(["Member_number", "Date"])["itemDescription"].nunique()
    sns.histplot(basket_sizes, bins=range(1, basket_sizes.max() + 2), kde=False, color="#2b5c8f", ax=axes[1])
    axes[1].axvline(basket_sizes.mean(), color="crimson", linestyle="--", linewidth=2, label=f"Mean = {basket_sizes.mean():.2f}")
    axes[1].set_title("Distribution of Basket Sizes (Items per Transaction)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Unique Items in Basket")
    axes[1].set_ylabel("Number of Transactions")
    axes[1].legend()

    plt.tight_layout()
    save_path = output_dir / "01_eda_top_items_and_basket_size.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_performance_comparison(benchmark_df: pd.DataFrame, output_dir: Path) -> Path:
    """Vẽ biểu đồ đôi so sánh Runtime (ms) và Peak Memory (MB) của Apriori vs FP-Growth."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    x_labels = [str(s) for s in benchmark_df["min_support"]]
    x = np.arange(len(x_labels))
    width = 0.36

    # 1. So sánh Runtime (ms)
    rects1 = axes[0].bar(x - width / 2, benchmark_df["apriori_runtime_ms"], width, label="Apriori", color="#e05a47")
    rects2 = axes[0].bar(x + width / 2, benchmark_df["fpgrowth_runtime_ms"], width, label="FP-Growth", color="#2a9d8f")
    axes[0].set_title("Execution Runtime Comparison: Apriori vs. FP-Growth", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Minimum Support Threshold (min_support)")
    axes[0].set_ylabel("Execution Time (ms)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels)
    axes[0].legend()
    axes[0].bar_label(rects1, fmt="%.1f", padding=2, fontsize=8)
    axes[0].bar_label(rects2, fmt="%.1f", padding=2, fontsize=8)

    # 2. So sánh Peak Memory (MB)
    rects3 = axes[1].bar(x - width / 2, benchmark_df["apriori_peak_mem_mb"], width, label="Apriori", color="#f4a261")
    rects4 = axes[1].bar(x + width / 2, benchmark_df["fpgrowth_peak_mem_mb"], width, label="FP-Growth", color="#264653")
    axes[1].set_title("Peak Memory Usage Comparison: Apriori vs. FP-Growth", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Minimum Support Threshold (min_support)")
    axes[1].set_ylabel("Peak Memory Usage (MB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels)
    axes[1].legend()
    axes[1].bar_label(rects3, fmt="%.2f", padding=2, fontsize=8)
    axes[1].bar_label(rects4, fmt="%.2f", padding=2, fontsize=8)

    plt.tight_layout()
    save_path = output_dir / "02_performance_apriori_vs_fpgrowth.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_support_confidence_heatmap(grid_df: pd.DataFrame, output_dir: Path) -> Path:
    """Vẽ Heatmap số lượng luật sinh ra và số luật Lift > 1 theo lưới Support - Confidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    pivot_total = grid_df.pivot(index="min_support", columns="min_confidence", values="total_rules")
    pivot_lift = grid_df.pivot(index="min_support", columns="min_confidence", values="rules_lift_gt_1")

    sns.heatmap(pivot_total, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar_kws={"label": "Number of Rules"})
    axes[0].set_title("Total Association Rules Generated", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Minimum Confidence Threshold")
    axes[0].set_ylabel("Minimum Support Threshold")

    sns.heatmap(pivot_lift, annot=True, fmt="d", cmap="YlOrRd", ax=axes[1], cbar_kws={"label": "Rules with Lift > 1"})
    axes[1].set_title("Filtered Rules with Lift > 1", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Minimum Confidence Threshold")
    axes[1].set_ylabel("Minimum Support Threshold")

    plt.tight_layout()
    save_path = output_dir / "03_support_confidence_heatmap.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_rules_distribution(rules_df: pd.DataFrame, output_dir: Path, top_n_network: int = 5) -> Path:
    """Vẽ Scatter plot (Support vs Confidence, màu theo Lift) và đồ thị mạng luật kết hợp Lift > 1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if rules_df.empty:
        return output_dir

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # 1. Scatter Plot: Support vs Confidence (colored by Lift)
    rules_gt1 = rules_df[rules_df["lift"] > 1.0]
    rules_le1 = rules_df[rules_df["lift"] <= 1.0]

    if not rules_le1.empty:
        axes[0].scatter(
            rules_le1["support"],
            rules_le1["confidence"],
            c="lightgray",
            alpha=0.6,
            s=40,
            label="Lift <= 1.0",
        )

    scatter = axes[0].scatter(
        rules_gt1["support"],
        rules_gt1["confidence"],
        c=rules_gt1["lift"],
        cmap="plasma",
        alpha=0.85,
        s=70,
        edgecolors="k",
        linewidth=0.4,
        label="Lift > 1.0",
    )
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label("Lift Value")
    axes[0].set_title("Association Rules: Support vs. Confidence (Colored by Lift)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Support")
    axes[0].set_ylabel("Confidence")
    axes[0].legend(loc="upper right")

    # 2. Network Graph cho Top luật có Lift > 1 cao nhất
    top_rules = rules_gt1.sort_values(by="lift", ascending=False).head(top_n_network)
    G = nx.DiGraph()
    for _, row in top_rules.iterrows():
        ant = row["antecedents_str"]
        con = row["consequents_str"]
        G.add_edge(ant, con, weight=row["lift"], confidence=row["confidence"])

    pos = nx.spring_layout(G, k=1.1, seed=42)
    edges = G.edges(data=True)
    weights = [d["weight"] * 1.8 for (_, _, d) in edges]

    nx.draw_networkx_nodes(G, pos, ax=axes[1], node_size=900, node_color="#a8dadc", edgecolors="#1d3557", linewidths=1.5)
    nx.draw_networkx_edges(
        G,
        pos,
        ax=axes[1],
        width=weights,
        edge_color="#e63946",
        arrowstyle="-|>",
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_labels(G, pos, ax=axes[1], font_size=8.5, font_weight="bold", font_color="#1d3557")

    axes[1].set_title(f"Top {len(top_rules)} High-Lift Association Rules Network (Lift > 1)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    save_path = output_dir / "04_rules_scatter_and_network.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
