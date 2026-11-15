"""
Module gom nhóm giao dịch và chuyển đổi sang dạng ma trận (Matrix Format).
"""

import pandas as pd
from typing import List
from mlxtend.preprocessing import TransactionEncoder


def group_transactions(
    df: pd.DataFrame,
    group_cols: List[str] = ["Member_number", "Date"],
) -> List[List[str]]:
    """
    Gom nhóm các mặt hàng được mua trong cùng một lượt (cùng khách hàng, cùng ngày)
    thành danh sách các giỏ hàng (List of Lists). Loại bỏ trùng lặp món trong cùng 1 giỏ.
    """
    transactions = (
        df.groupby(group_cols)["itemDescription"]
        .apply(lambda items: sorted(list(set(items))))
        .tolist()
    )
    return transactions


def create_basket_matrix(transactions: List[List[str]]) -> pd.DataFrame:
    """
    Chuyển đổi danh sách giỏ hàng sang dạng ma trận nhị phân (Boolean Matrix)
    sử dụng TransactionEncoder của mlxtend.
    """
    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    basket_matrix = pd.DataFrame(encoded_array, columns=encoder.columns_, dtype=bool)
    return basket_matrix
