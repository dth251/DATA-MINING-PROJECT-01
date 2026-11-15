"""
Module đọc dữ liệu thô, làm sạch và thống kê khám phá (EDA).
"""

import pandas as pd
from pathlib import Path


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Đọc và làm sạch sơ bộ dữ liệu giao dịch gốc Groceries_dataset.csv."""
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc tại: {filepath}")

    df = pd.read_csv(filepath)

    # Chuẩn hóa tên cột và dữ liệu chuỗi
    df.columns = [col.strip() for col in df.columns]
    df["itemDescription"] = df["itemDescription"].astype(str).str.strip().str.lower()
    df["Date"] = df["Date"].astype(str).str.strip()

    # Loại bỏ các dòng bị thiếu giá trị quan trọng
    df = df.dropna(subset=["Member_number", "Date", "itemDescription"]).reset_index(drop=True)
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """Thống kê tổng quan về số lượng khách hàng, ngày, mặt hàng và giao dịch."""
    total_rows = len(df)
    exact_duplicates = int(df.duplicated().sum())
    unique_members = int(df["Member_number"].nunique())
    unique_items = int(df["itemDescription"].nunique())
    unique_dates = int(df["Date"].nunique())

    # Số giỏ hàng khi gom theo (Member_number, Date)
    basket_sizes = df.groupby(["Member_number", "Date"])["itemDescription"].nunique()
    total_baskets = len(basket_sizes)
    avg_basket_size = float(basket_sizes.mean())
    max_basket_size = int(basket_sizes.max())
    min_basket_size = int(basket_sizes.min())

    stats = {
        "total_records": total_rows,
        "exact_duplicate_rows": exact_duplicates,
        "unique_members": unique_members,
        "unique_dates": unique_dates,
        "unique_items": unique_items,
        "total_transactions": total_baskets,
        "avg_items_per_transaction": round(avg_basket_size, 2),
        "min_items_per_transaction": min_basket_size,
        "max_items_per_transaction": max_basket_size,
    }
    return stats
