"""
Module lọc các luật kết hợp có Lift > 1 và đề xuất chiến lược Bán chéo (Cross-selling).
"""

from typing import List, Tuple

import pandas as pd

from src.algorithms.association_rules import generate_rules


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


def filter_lift_rules_by_confidence(
    frequent_itemsets: pd.DataFrame,
    confidence_list: List[float],
    min_lift: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sinh luật kết hợp tại từng ngưỡng min_confidence và lọc các luật có Lift > min_lift.

    Trả về:
    - summary_df: mỗi dòng là một ngưỡng min_confidence (tổng số luật, số luật Lift > min_lift,
      tỷ lệ phần trăm, Lift lớn nhất và trung bình).
    - rules_df: toàn bộ luật Lift > min_lift của mọi ngưỡng, cột `min_confidence` cho biết ngưỡng.
      Luật đạt ngưỡng cao cũng xuất hiện ở các ngưỡng thấp hơn.
    """
    rule_cols = ["rule_str", "antecedents_str", "consequents_str", "support", "confidence", "lift", "leverage", "conviction"]
    summary_records = []
    rule_frames = []

    for min_conf in confidence_list:
        rules = generate_rules(frequent_itemsets, min_confidence=min_conf)
        lifted = filter_positive_lift_rules(rules, min_lift=min_lift)
        n_rules, n_lifted = len(rules), len(lifted)

        summary_records.append(
            {
                "min_confidence": min_conf,
                "total_rules": n_rules,
                "rules_lift_gt_min": n_lifted,
                "pct_lift_gt_min": round(n_lifted / n_rules * 100, 2) if n_rules else 0.0,
                "max_lift": round(float(lifted["lift"].max()), 4) if n_lifted else 0.0,
                "mean_lift": round(float(lifted["lift"].mean()), 4) if n_lifted else 0.0,
            }
        )
        if n_lifted:
            frame = lifted[rule_cols].copy()
            frame.insert(0, "min_confidence", min_conf)
            rule_frames.append(frame)

    summary_df = pd.DataFrame(summary_records)
    rules_df = pd.concat(rule_frames, ignore_index=True) if rule_frames else pd.DataFrame(columns=["min_confidence"] + rule_cols)
    return summary_df, rules_df


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
