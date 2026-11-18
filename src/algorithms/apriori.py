"""
Module khai phá tập mặt hàng phổ biến (Frequent Itemsets) bằng thuật toán Apriori.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori


def run_apriori(basket_matrix: pd.DataFrame, min_support: float) -> pd.DataFrame:
    """Thực thi thuật toán Apriori tìm tập mặt hàng phổ biến."""
    freq_items = apriori(basket_matrix, min_support=min_support, use_colnames=True)
    if not freq_items.empty:
        freq_items["length"] = freq_items["itemsets"].apply(len)
        freq_items = freq_items.sort_values(by=["support", "length"], ascending=[False, True]).reset_index(drop=True)
    return freq_items
