"""
Entry-point chạy toàn bộ pipeline cho bài toán:
I. Pattern Mining - Supermarket Market Basket Analysis
"""

import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from src.config import (
    RAW_DATA_FILE,
    DATA_PROCESSED_DIR,
    MATRIX_OUTPUT_FILE,
    OUTPUT_FIGURES_DIR,
    OUTPUT_TABLES_DIR,
    SUPPORT_THRESHOLDS,
    CONFIDENCE_THRESHOLDS,
    DEFAULT_MIN_SUPPORT,
    DEFAULT_MIN_CONFIDENCE,
    MIN_LIFT_THRESHOLD,
)
from src.data_processing import (
    load_raw_data,
    explore_data,
    group_transactions,
    create_basket_matrix,
)
from src.algorithms import (
    mine_frequent_itemsets,
    generate_rules,
    evaluate_support_confidence_grid,
)
from src.evaluation import compare_algorithms_performance
from src.business_strategy import (
    filter_positive_lift_rules,
    filter_lift_rules_by_confidence,
    recommend_cross_selling,
    recommend_store_layout,
)
from src.visualization import (
    plot_performance_comparison,
    plot_support_confidence_heatmap,
    plot_rules_distribution,
)


def main() -> None:
    """Hàm điều phối luồng chạy chính của dự án."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("I. PATTERN MINING: SUPERMARKET MARKET BASKET ANALYSIS PIPELINE")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Bước 1: Đọc dữ liệu, EDA & Chuyển đổi sang dạng ma trận (Matrix Format)
    # -------------------------------------------------------------------------
    print("\n[1/5] Đang đọc dữ liệu gốc và chuyển đổi sang dạng ma trận (Matrix Format)...")
    df_raw = load_raw_data(RAW_DATA_FILE)
    eda_stats = explore_data(df_raw)
    for k, v in eda_stats.items():
        print(f"  - {k}: {v}")

    transactions = group_transactions(df_raw)
    basket_matrix = create_basket_matrix(transactions)
    sparsity = (1.0 - basket_matrix.values.mean()) * 100
    print(f"  - Kích thước ma trận giỏ hàng (Boolean Matrix): {basket_matrix.shape} (Độ thưa: {sparsity:.2f}%)")

    # Lưu ma trận và thống kê EDA
    basket_matrix.to_csv(MATRIX_OUTPUT_FILE, index=False)
    pd.DataFrame([eda_stats]).to_csv(OUTPUT_TABLES_DIR / "01_eda_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # Bước 2: Thực thi Apriori & FP-Growth với nhiều ngưỡng Support/Confidence
    # -------------------------------------------------------------------------
    print("\n[2/5] Đang thực thi Apriori & FP-Growth trên lưới ngưỡng Support/Confidence...")
    grid_df = evaluate_support_confidence_grid(
        basket_matrix,
        support_list=SUPPORT_THRESHOLDS,
        confidence_list=CONFIDENCE_THRESHOLDS,
    )
    grid_df.to_csv(OUTPUT_TABLES_DIR / "02_support_confidence_grid.csv", index=False)
    print(grid_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Bước 3: So sánh hiệu năng thực thi (Runtime & Memory Usage)
    # -------------------------------------------------------------------------
    print("\n[3/5] Đang đo lường và so sánh hiệu năng (Runtime & Peak Memory)...")
    benchmark_df = compare_algorithms_performance(
        basket_matrix,
        support_thresholds=SUPPORT_THRESHOLDS,
    )
    benchmark_df.to_csv(OUTPUT_TABLES_DIR / "03_benchmark_apriori_vs_fpgrowth.csv", index=False)
    print(benchmark_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Bước 4: Lọc luật có Lift > 1 & Đề xuất chiến lược Cross-selling / Store Layout
    # -------------------------------------------------------------------------
    print(
        f"\n[4/5] Đang trích xuất luật kết hợp tại min_support={DEFAULT_MIN_SUPPORT}, "
        f"min_confidence={DEFAULT_MIN_CONFIDENCE} và lọc Lift > {MIN_LIFT_THRESHOLD}..."
    )
    freq_items = mine_frequent_itemsets(basket_matrix, min_support=DEFAULT_MIN_SUPPORT, algorithm="fpgrowth")
    all_rules = generate_rules(freq_items, min_confidence=DEFAULT_MIN_CONFIDENCE)
    rules_lift_gt1 = filter_positive_lift_rules(all_rules, min_lift=MIN_LIFT_THRESHOLD)

    print(f"  - Tổng số tập phổ biến tìm được: {len(freq_items)}")
    print(f"  - Tổng số luật sinh ra: {len(all_rules)}")
    print(f"  - Số luật có Lift > {MIN_LIFT_THRESHOLD}: {len(rules_lift_gt1)}")

    cross_sell_df = recommend_cross_selling(rules_lift_gt1, top_n=15)
    store_layout_df = recommend_store_layout(rules_lift_gt1, top_n=15)

    rules_lift_gt1.to_csv(OUTPUT_TABLES_DIR / "04_filtered_rules_lift_gt_1.csv", index=False)
    cross_sell_df.to_csv(OUTPUT_TABLES_DIR / "05_cross_selling_recommendations.csv", index=False)
    store_layout_df.to_csv(OUTPUT_TABLES_DIR / "06_store_layout_recommendations.csv", index=False)

    print("\n--- TOP 10 LUẬT KẾT HỢP CÓ LIFT > 1 MẠNH NHẤT ---")
    display_cols = ["rule_str", "support", "confidence", "lift", "conviction"]
    print(rules_lift_gt1[display_cols].head(10).to_string(index=False))

    print(
        f"\n--- LỌC LUẬT LIFT > {MIN_LIFT_THRESHOLD} THEO TỪNG NGƯỠNG MIN_CONFIDENCE "
        f"(min_support={DEFAULT_MIN_SUPPORT}) ---"
    )
    conf_summary_df, conf_rules_df = filter_lift_rules_by_confidence(
        freq_items,
        confidence_list=CONFIDENCE_THRESHOLDS,
        min_lift=MIN_LIFT_THRESHOLD,
    )
    conf_summary_df.to_csv(OUTPUT_TABLES_DIR / "07_lift_rules_summary_by_confidence.csv", index=False)
    conf_rules_df.to_csv(OUTPUT_TABLES_DIR / "08_lift_rules_by_confidence.csv", index=False)
    print(conf_summary_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Bước 5: Trực quan hóa dữ liệu và kết quả thực nghiệm
    # -------------------------------------------------------------------------
    print("\n[5/5] Đang xuất các biểu đồ trực quan hóa vào outputs/figures/...")
    fig2 = plot_performance_comparison(benchmark_df, OUTPUT_FIGURES_DIR)
    fig3 = plot_support_confidence_heatmap(grid_df, OUTPUT_FIGURES_DIR)
    fig4 = plot_rules_distribution(all_rules, OUTPUT_FIGURES_DIR)

    print(f"  - Đã lưu: {fig2.name}")
    print(f"  - Đã lưu: {fig3.name}")
    print(f"  - Đã lưu: {fig4.name}")
    print("\nHoàn tất toàn bộ pipeline!")


if __name__ == "__main__":
    main()
