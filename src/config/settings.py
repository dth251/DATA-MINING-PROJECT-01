"""
Module cấu hình tham số, đường dẫn thư mục và các ngưỡng thực nghiệm
cho bài toán Supermarket Market Basket Analysis.
"""

import math
from pathlib import Path

# Đường dẫn gốc của dự án (assignment/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Đường dẫn dữ liệu
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_FILE = DATA_RAW_DIR / "Groceries_dataset.csv"
MATRIX_OUTPUT_FILE = DATA_PROCESSED_DIR / "basket_matrix.csv"

# Đường dẫn xuất kết quả
OUTPUT_FIGURES_DIR = BASE_DIR / "outputs" / "figures"
OUTPUT_TABLES_DIR = BASE_DIR / "outputs" / "tables"

# Các ngưỡng thực nghiệm cho Apriori và FP-Growth, khai báo theo số giỏ hàng tối thiểu
# (min_support = số giỏ / tổng số giỏ). Tổng số giỏ (Member_number + Date) của bộ dữ liệu là 14.963.
N_TRANSACTIONS = 14963
SUPPORT_BASKET_COUNTS = [10, 25, 50, 75, 100, 200]
# Làm tròn xuống 6 chữ số thập phân để itemset xuất hiện đúng `count` giỏ vẫn đạt ngưỡng
SUPPORT_THRESHOLDS = [math.floor(c / N_TRANSACTIONS * 1e6) / 1e6 for c in SUPPORT_BASKET_COUNTS]
CONFIDENCE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]

# Ngưỡng mặc định để phân tích sâu tập luật kết hợp
DEFAULT_MIN_SUPPORT = 0.001
DEFAULT_MIN_CONFIDENCE = 0.10
MIN_LIFT_THRESHOLD = 1.0

