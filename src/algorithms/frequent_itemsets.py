"""
Module điều phối chạy Apriori (apriori.py) hoặc FP-Growth (fpgrowth.py) theo tên thuật toán.
"""

import pandas as pd

from .apriori import run_apriori
from .fpgrowth import run_fpgrowth


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
