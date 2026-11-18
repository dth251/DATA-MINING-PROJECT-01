"""
Kiểm thử tự động (Unit Tests) cho toàn bộ 4 nhiệm vụ của dự án
Supermarket Market Basket Analysis.
"""

import unittest
from src.config import RAW_DATA_FILE
from src.data_processing import load_raw_data, group_transactions, create_basket_matrix
from src.algorithms import run_apriori, run_fpgrowth, generate_rules
from src.business_strategy import filter_positive_lift_rules, filter_lift_rules_by_confidence


class TestMarketBasketPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_raw_data(RAW_DATA_FILE)
        cls.transactions = group_transactions(cls.df)
        cls.matrix = create_basket_matrix(cls.transactions)

    def test_matrix_dimensions(self):
        """Kiểm tra kích thước ma trận nhị phân (14963 giỏ hàng x 167 mặt hàng)."""
        self.assertEqual(self.matrix.shape, (14963, 167))
        self.assertEqual(self.matrix.dtypes.unique().tolist(), [bool])

    def test_apriori_fpgrowth_consistency(self):
        """Kiểm tra Apriori và FP-Growth trả về cùng số lượng tập phổ biến."""
        ap_items = run_apriori(self.matrix, min_support=0.002)
        fp_items = run_fpgrowth(self.matrix, min_support=0.002)
        self.assertEqual(len(ap_items), len(fp_items))
        self.assertEqual(len(fp_items), 330)

    def test_rules_and_lift_filtering(self):
        """Kiểm tra lọc luật có Lift > 1."""
        fp_items = run_fpgrowth(self.matrix, min_support=0.001)
        rules = generate_rules(fp_items, min_confidence=0.10)
        lift_gt1 = filter_positive_lift_rules(rules, min_lift=1.0)
        self.assertFalse(lift_gt1.empty)
        self.assertTrue((lift_gt1["lift"] > 1.0).all())

    def test_lift_filtering_by_confidence(self):
        """Kiểm tra lọc Lift > 1 theo nhiều ngưỡng min_confidence."""
        fp_items = run_fpgrowth(self.matrix, min_support=0.001)
        conf_list = [0.05, 0.10, 0.15, 0.20]
        summary, rules = filter_lift_rules_by_confidence(fp_items, conf_list, min_lift=1.0)
        self.assertEqual(summary["min_confidence"].tolist(), conf_list)
        self.assertEqual(summary["rules_lift_gt_min"].tolist(), [94, 37, 8, 2])
        self.assertTrue((rules["lift"] > 1.0).all())
        self.assertTrue((rules["confidence"] >= rules["min_confidence"]).all())


if __name__ == "__main__":
    unittest.main()

