"""
Module lọc các luật kết hợp có Lift > 1 và đề xuất chiến lược Bán chéo (Cross-selling).
"""

import pandas as pd


def filter_positive_lift_rules(
    rules: pd.DataFrame,
    min_lift: float = 1.0,
) -> pd.DataFrame:
    """Lọc các luật kết hợp có Lift > min_lift và sắp xếp giảm dần theo Lift, Confidence."""
    if rules.empty:
        return pd.DataFrame()

    filtered = rules[rules["lift"] > min_lift].copy()
    filtered = filtered.sort_values(by=["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return filtered


def recommend_cross_selling(rules_lift_gt1: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Trích xuất các luật có Lift > 1 và Confidence cao nhất để gợi ý:
    - Bán chéo (Cross-selling)
    - Đóng gói sản phẩm khuyến mãi (Product Bundling)
    """
    if rules_lift_gt1.empty:
        return pd.DataFrame()

    cols = [
        "antecedents_str",
        "consequents_str",
        "rule_str",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
    ]
    available_cols = [c for c in cols if c in rules_lift_gt1.columns]
    top_rules = rules_lift_gt1[available_cols].head(top_n).copy()

    # Gán nhãn gợi ý chiến lược bán chéo cụ thể cho từng luật
    recommendations = []
    for _, row in top_rules.iterrows():
        ant = row["antecedents_str"]
        con = row["consequents_str"]
        conf_pct = row["confidence"] * 100
        lift_val = row["lift"]
        if "," in ant:
            rec = (
                f"Combo/Bundle: Khi khách mua [{ant}], có {conf_pct:.1f}% khả năng mua thêm [{con}] "
                f"(Lift={lift_val:.2f}). Đề xuất gói combo giảm giá 5-10%."
            )
        else:
            rec = (
                f"Cross-sell/Voucher: Khi khách chọn [{ant}], gợi ý mua kèm [{con}] "
                f"(Confidence={conf_pct:.1f}%, Lift={lift_val:.2f})."
            )
        recommendations.append(rec)

    top_rules["business_action"] = recommendations
    return top_rules
