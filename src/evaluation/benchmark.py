"""
Module đánh giá và so sánh hiệu năng thực thi (Runtime & Memory Usage)
giữa hai thuật toán Apriori và FP-Growth.
"""

import gc
import time
import tracemalloc
import pandas as pd
from typing import List, Tuple
from src.algorithms.frequent_itemsets import mine_frequent_itemsets


def measure_execution(
    basket_matrix: pd.DataFrame,
    min_support: float,
    algorithm: str,
    n_repeats: int = 3,
) -> Tuple[float, float, int]:
    """
    Đo thời gian thực thi trung bình (ms) và bộ nhớ đỉnh (Peak Memory - MB)
    của một thuật toán tại ngưỡng min_support xác định.
    Tách riêng bước đo Runtime và bước đo Memory để tránh overhead của tracemalloc
    làm sai lệch thời gian chạy thực tế của cấu trúc cây FP-Tree.
    """
    runtimes_ms = []
    itemset_count = 0

    # 1. Đo thời gian chạy thực tế (Runtime) không bị nhiễu bởi tracemalloc
    for _ in range(n_repeats):
        gc.collect()
        start_time = time.perf_counter()
        freq_items = mine_frequent_itemsets(basket_matrix, min_support=min_support, algorithm=algorithm)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        runtimes_ms.append(elapsed_ms)
        itemset_count = len(freq_items)

    # 2. Đo dung lượng bộ nhớ đỉnh (Peak Memory Usage) bằng tracemalloc
    gc.collect()
    tracemalloc.start()
    _ = mine_frequent_itemsets(basket_matrix, min_support=min_support, algorithm=algorithm)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_runtime_ms = sum(runtimes_ms) / len(runtimes_ms)
    peak_mb = peak_bytes / (1024.0 * 1024.0)
    return round(avg_runtime_ms, 2), round(peak_mb, 3), itemset_count


def compare_algorithms_performance(
    basket_matrix: pd.DataFrame,
    support_thresholds: List[float],
) -> pd.DataFrame:
    """
    Chạy thực nghiệm so sánh Apriori và FP-Growth trên dải ngưỡng min_support.
    Trả về DataFrame gồm Runtime (ms), Peak Memory (MB) và tỷ lệ chênh lệch.
    """
    results = []

    for min_sup in support_thresholds:
        ap_time, ap_mem, ap_count = measure_execution(basket_matrix, min_sup, "apriori")
        fp_time, fp_mem, fp_count = measure_execution(basket_matrix, min_sup, "fpgrowth")

        speedup = round(ap_time / fp_time, 2) if fp_time > 0 else 1.0
        mem_ratio = round(ap_mem / fp_mem, 2) if fp_mem > 0 else 1.0

        results.append(
            { 
                "min_support": min_sup,
                "frequent_itemsets": fp_count,
                "apriori_runtime_ms": ap_time,
                "fpgrowth_runtime_ms": fp_time,
                "speedup_ratio (Apriori/FP)": speedup,
                "apriori_peak_mem_mb": ap_mem,
                "fpgrowth_peak_mem_mb": fp_mem,
                "memory_ratio (Apriori/FP)": mem_ratio,
            }
        )

    return pd.DataFrame(results)
