"""
Module khai phá tập mặt hàng phổ biến (Frequent Itemsets) bằng Apriori và FP-Growth.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth


def run_apriori(basket_matrix: pd.DataFrame, min_support: float) -> pd.DataFrame:
    """Thực thi thuật toán Apriori tìm tập mặt hàng phổ biến."""
    freq_items = apriori(basket_matrix, min_support=min_support, use_colnames=True)
    if not freq_items.empty:
        freq_items["length"] = freq_items["itemsets"].apply(len)
        freq_items = freq_items.sort_values(by=["support", "length"], ascending=[False, True]).reset_index(drop=True)
    return freq_items


def run_fpgrowth(basket_matrix: pd.DataFrame, min_support: float) -> pd.DataFrame:
    """Thực thi thuật toán FP-Growth tìm tập mặt hàng phổ biến."""
    freq_items = fpgrowth(basket_matrix, min_support=min_support, use_colnames=True)
    if not freq_items.empty:
        freq_items["length"] = freq_items["itemsets"].apply(len)
        freq_items = freq_items.sort_values(by=["support", "length"], ascending=[False, True]).reset_index(drop=True)
    return freq_items


def mine_frequent_itemsets(
    basket_matrix: pd.DataFrame,
    min_support: float,
    algorithm: str = "fpgrowth",
) -> pd.DataFrame:
    """Hàm điều phối chạy Apriori hoặc FP-Growth theo tên thuật toán."""
    algo = algorithm.strip().lower()
    if algo == "apriori":
        return run_apriori(basket_matrix, min_support)
    elif algo in ("fpgrowth", "fp-growth"):
        return run_fpgrowth(basket_matrix, min_support)
    else:
        raise ValueError(f"Thuật toán không hợp lệ: {algorithm}. Chọn 'apriori' hoặc 'fpgrowth'.")
