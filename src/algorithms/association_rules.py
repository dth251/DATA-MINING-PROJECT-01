"""
Module sinh luật kết hợp (Association Rules) và thực nghiệm đa ngưỡng Support/Confidence.
"""

import pandas as pd
from typing import List
from mlxtend.frequent_patterns import association_rules
from .apriori import run_apriori
from .fpgrowth import run_fpgrowth


def format_itemset(itemset: frozenset) -> str:
    """Chuyển frozenset thành chuỗi dễ đọc, ví dụ: 'whole milk, yogurt'."""
    return ", ".join(sorted(list(itemset)))


def generate_rules(
    frequent_itemsets: pd.DataFrame,
    min_confidence: float,
) -> pd.DataFrame:
    """Sinh luật kết hợp từ tập phổ biến theo ngưỡng min_confidence."""
    if frequent_itemsets.empty:
        return pd.DataFrame()

    # Kiểm tra xem có tập phổ biến nào chứa từ 2 phần tử trở lên không
    has_multi_items = frequent_itemsets["itemsets"].apply(len).max() >= 2
    if not has_multi_items:
        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )
    if not rules.empty:
        rules["antecedents_str"] = rules["antecedents"].apply(format_itemset)
        rules["consequents_str"] = rules["consequents"].apply(format_itemset)
        rules["rule_str"] = rules["antecedents_str"] + " -> " + rules["consequents_str"]
        rules = rules.sort_values(by=["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return rules


def evaluate_support_confidence_grid(
    basket_matrix: pd.DataFrame,
    support_list: List[float],
    confidence_list: List[float],
) -> pd.DataFrame:
    """
    Thực nghiệm lưới các ngưỡng Support và Confidence khác nhau,
    thống kê số lượng tập phổ biến (1-itemset, >=2-itemsets), số lượng luật sinh ra
    và số lượng luật có Lift > 1.
    Đồng thời kiểm chứng kết quả giữa Apriori và FP-Growth.
    """
    records = []

    for min_sup in support_list:
        apriori_sets = run_apriori(basket_matrix, min_support=min_sup)
        fpgrowth_sets = run_fpgrowth(basket_matrix, min_support=min_sup)

        total_itemsets = len(fpgrowth_sets)
        multi_itemsets = int((fpgrowth_sets["length"] >= 2).sum()) if not fpgrowth_sets.empty else 0
        is_identical = len(apriori_sets) == len(fpgrowth_sets)

        for min_conf in confidence_list:
            rules = generate_rules(fpgrowth_sets, min_confidence=min_conf)
            total_rules = len(rules)
            lift_gt1_rules = int((rules["lift"] > 1.0).sum()) if not rules.empty else 0
            max_lift = float(rules["lift"].max()) if not rules.empty else 0.0

            records.append(
                {
                    "min_support": min_sup,
                    "min_confidence": min_conf,
                    "total_frequent_itemsets": total_itemsets,
                    "itemsets_len_ge_2": multi_itemsets,
                    "apriori_fpgrowth_match": is_identical,
                    "total_rules": total_rules,
                    "rules_lift_gt_1": lift_gt1_rules,
                    "max_lift": round(max_lift, 4),
                }
            )

    return pd.DataFrame(records)
