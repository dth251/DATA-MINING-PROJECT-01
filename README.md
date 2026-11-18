# I. Pattern Mining - Supermarket Market Basket Analysis

Dự án khai phá mẫu phổ biến và luật kết hợp trên bộ dữ liệu **Groceries Market Basket Dataset** sử dụng hai thuật toán **Apriori** và **FP-Growth**.

## 1. Cấu trúc thư mục

```text
assignment/
├── data/
│   ├── raw/
│   │   └── Groceries_dataset.csv        # Dữ liệu gốc (38,765 bản ghi)
│   └── processed/
│       └── basket_matrix.csv            # Ma trận nhị phân Boolean (14,963 × 167)
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                  # Cấu hình đường dẫn & ngưỡng Support/Confidence/Lift
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── loader.py                    # Đọc dữ liệu thô, làm sạch & thống kê EDA
│   │   └── matrix_builder.py            # Gom nhóm giỏ hàng & mã hóa ma trận nhị phân
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── apriori.py                   # Thực thi thuật toán Apriori
│   │   ├── fpgrowth.py                  # Thực thi thuật toán FP-Growth
│   │   ├── frequent_itemsets.py         # Điều phối chạy Apriori hoặc FP-Growth theo tên
│   │   └── association_rules.py         # Sinh luật kết hợp đa ngưỡng Support/Confidence
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── benchmark.py                 # Đo và so sánh Runtime (ms) & Peak Memory (MB)
│   ├── business_strategy/
│   │   ├── __init__.py
│   │   ├── cross_selling.py             # Lọc luật Lift > 1 & gợi ý bán chéo / combo
│   │   └── store_layout.py              # Gợi ý bố trí quầy kệ siêu thị (Store Layout)
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                     # Vẽ biểu đồ EDA, hiệu năng, Heatmap & Network
├── notebooks/
│   └── market_basket_analysis.ipynb     # Notebook trình bày kết quả thực nghiệm trực quan
├── outputs/
│   ├── figures/                         # Chứa 4 biểu đồ (.png) phục vụ báo cáo
│   └── tables/                          # Chứa 6 bảng thống kê & tập luật (.csv)
├── tests/
│   └── test_pipeline.py                 # Kiểm thử tự động (Unit Tests)
├── main.py                              # Kịch bản chính chạy toàn bộ pipeline
├── REPORT.md                            # Báo cáo thực nghiệm chi tiết
├── requirements.txt                     # Các thư viện Python cần thiết
└── README.md                            # Hướng dẫn sử dụng dự án
```

## 2. Hướng dẫn chạy dự án

Mở Terminal (PowerShell hoặc Command Prompt) tại thư mục gốc của dự án (`c:\Kì 1 Năm 4\Khai phá dữ liệu\assignment`):

### Cách 1: Chạy toàn bộ Pipeline tự động (Khuyên dùng)
```powershell
python main.py
```
Lệnh này sẽ tự động chạy từ Bước 1 đến Bước 5, in kết quả trực tiếp ra màn hình Terminal và xuất toàn bộ bảng biểu vào `outputs/tables/` cùng biểu đồ vào `outputs/figures/`.

### Cách 2: Xem và chạy trên Jupyter Notebook
Mở trực tiếp file `notebooks/market_basket_analysis.ipynb` trong VS Code / Antigravity IDE (các bảng và biểu đồ đã được chạy sẵn bên trong), hoặc khởi chạy Jupyter:
```powershell
jupyter notebook notebooks/market_basket_analysis.ipynb
```

### Cách 3: Chạy bộ kiểm thử tự động (Unit Tests)
```powershell
python -m unittest discover -s tests -v
```

### Tùy chỉnh tham số thực nghiệm
Nếu muốn thay đổi dải ngưỡng `min_support`, `min_confidence` hoặc `min_lift`, bạn chỉ cần chỉnh sửa trong file `src/config/settings.py` rồi chạy lại `python main.py`.
