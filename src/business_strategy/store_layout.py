"""
Module phân tích tập luật có Lift > 1 để đề xuất chiến lược bố trí quầy kệ (Store Layout).
"""

import pandas as pd


def recommend_store_layout(rules_lift_gt1: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Phân nhóm các cặp mặt hàng có Lift > 1 phục vụ bố trí quầy kệ (Store Layout):
    1. Kệ đặt cạnh nhau (Adjacent / Complementary Placement):
       Các luật có Lift cao nhất -> đặt gần nhau để kích thích mua hàng tiện lợi/bổ trợ.
    2. Kệ đặt xa nhau / Cuối hành lang (Anchor / Traffic-Flow Placement):
       Các luật có cả Support cao và Lift > 1 (các mặt hàng thiết yếu đi cùng nhau) ->
       đặt ở các khu vực cách nhau để kéo khách hàng đi qua nhiều gian hàng trung gian.
    """
    if rules_lift_gt1.empty:
        return pd.DataFrame()

    df = rules_lift_gt1.copy()
    median_support = df["support"].median()

    layout_records = []
    for _, row in df.iterrows():
        ant = row["antecedents_str"]
        con = row["consequents_str"]
        sup = row["support"]
        conf = row["confidence"]
        lift = row["lift"]

        if sup >= median_support and conf >= 0.12:
            zone_strategy = "Anchor Placement (Đặt cách xa / Đầu-Cuối hành lang)"
            rationale = (
                f"Tần suất mua chung cao (Support={sup:.4f}, Lift={lift:.2f}). "
                f"Đặt [{ant}] và [{con}] ở hai đầu lối đi để kéo luồng khách di chuyển qua các kệ hàng biên lợi nhuận cao."
            )
        else:
            zone_strategy = "Complementary Co-location (Đặt kệ kế cận / Cross-merchandising)"
            rationale = (
                f"Độ liên kết bổ trợ mạnh (Lift={lift:.2f}, Confidence={conf*100:.1f}%). "
                f"Trưng bày [{con}] ngay cạnh quầy [{ant}] để kích thích mua ngẫu hứng (Impulse Buying)."
            )

        layout_records.append(
            {
                "antecedents": ant,
                "consequents": con,
                "support": round(sup, 5),
                "confidence": round(conf, 4),
                "lift": round(lift, 4),
                "layout_strategy": zone_strategy,
                "rationale": rationale,
            }
        )

    layout_df = pd.DataFrame(layout_records)
    return layout_df.head(top_n * 2)
