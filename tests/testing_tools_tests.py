"""
Unit-tests for functions used in testing only.

build_input_node_lists_and_truth_tables(), feature A: determining node names correctly.
0: no nodes are input to their update rule | 1: nodes are input to their update rule
0: no constants are used in update rules | 1: constants are used in update rules
COMPLETE.

generate_node_name(), feature A: generating node name without parameters provided.
COMPLETE.

generate_node_name(), feature B: generating node name of length in given range.
COMPLETE.

generate_node_name(), feature C: failing to generate node name for invalid range.
COMPLETE.
"""
import unittest

from boolsi.testing_tools import build_predecessor_nodes_lists_and_truth_tables, generate_node_name


class TestingToolsTestCase(unittest.TestCase):

    def test_build_input_node_lists_and_truth_tables_A_0_0(self):
        update_rule_dict = {'A': 'B ',
                            'B': 'not A'}

        expected_input_node_lists = [[1], [0]]
        expected_truth_tables = [{(False,): False, (True,): True},
                                 {(False,): True, (True,): False}]

        input_node_lists, truth_tables = build_predecessor_nodes_lists_and_truth_tables(update_rule_dict)

        self.assertEqual(input_node_lists, expected_input_node_lists)
        self.assertEqual(truth_tables, expected_truth_tables)

    def test_build_input_node_lists_and_truth_tables_A_0_1(self):
        update_rule_dict = {'A': '1 '}

        expected_input_node_lists = [[]]
        expected_truth_tables = [{(): True}]

        input_node_lists, truth_tables = build_predecessor_nodes_lists_and_truth_tables(update_rule_dict)

        self.assertEqual(input_node_lists, expected_input_node_lists)
        self.assertEqual(truth_tables, expected_truth_tables)

    def test_build_input_node_lists_and_truth_tables_A_1_0(self):
        update_rule_dict = {'A': 'not A '}

        expected_input_node_lists = [[0]]
        expected_truth_tables = [{(False,): True, (True,): False}]

        input_node_lists, truth_tables = build_predecessor_nodes_lists_and_truth_tables(update_rule_dict)

        self.assertEqual(input_node_lists, expected_input_node_lists)
        self.assertEqual(truth_tables, expected_truth_tables)

    def test_build_input_node_lists_and_truth_tables_A_1_1(self):
        update_rule_dict = {'A': '1 and not A'}

        expected_input_node_lists = [[0]]
        expected_truth_tables = [{(False,): True, (True,): False}]

        input_node_lists, truth_tables = build_predecessor_nodes_lists_and_truth_tables(update_rule_dict)

        self.assertEqual(input_node_lists, expected_input_node_lists)
        self.assertEqual(truth_tables, expected_truth_tables)

    def test_generate_node_name_A(self):
        node_name = generate_node_name()

        self.assertGreater(len(node_name), 0)

    def test_generate_node_name_B(self):
        length = 5

        node_name = generate_node_name(min_length=length, max_length=length)

        self.assertEqual(len(node_name), length)

    def test_generate_node_name_C(self):
        min_length = 5
        max_length = 4

        node_name = generate_node_name(min_length=min_length, max_length=max_length)

        self.assertIsNone(node_name)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
