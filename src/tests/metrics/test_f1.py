'''
This file implements tets for src/metrics/exact_match.py implementation.
'''

from unittest import TestCase

from src.metrics import compute_f1


class TestExactMatch(TestCase):

    def test_all_tokens_and_order_are_correct(self):
        '''
        Tests if the F1 for equal sentences is equal to 1.0.
        '''
        pred = '1 2 3 4 5 6 7 8 9 10'
        gold = '1 2 3 4 5 6 7 8 9 10'
        expected_f1 = 1.0

        self.assertEqual(compute_f1(gold, pred), expected_f1)

    def test_all_tokens_are_incorrect(self):
        '''
        Tests if the F1 for completely different sentences is equal to 0.0.
        '''
        pred = '1 2 3 4 5 6 7 8 9 10'
        gold = 'a b c d e f g h i j'
        expected_f1 = 0.0

        self.assertEqual(compute_f1(gold, pred), expected_f1)

    def test_is_order_independent(self):
        '''
        Tests if the F1 is order independent.
        '''
        pred = '1 2 3 4 5 6 7 8 9 10'
        gold = '10 8 6 4 2 9 7 5 3 1'
        expected_f1 = 1.0

        self.assertEqual(compute_f1(gold, pred), expected_f1)

    def test_empty_pred_and_empty_gold(self):
        '''
        Tests if the F1 is 1.0 for empty pred and gold sentences.
        '''
        pred = ''
        gold = ''
        expected_f1 = 1.0

        self.assertEqual(compute_f1(gold, pred), expected_f1)

    def test_empty_pred(self):
        '''
        Tests if the F1 is 0.0 for empty pred and not-empty gold.
        '''
        pred = ''
        gold = 'i am not empy'
        expected_f1 = 0.0

        self.assertEqual(compute_f1(gold, pred), expected_f1)

    def test_empty_gold(self):
        '''
        Tests if the F1 is 0.0 for empty gold and not-empty pred.
        '''
        pred = 'i am not empy'
        gold = ''
        expected_f1 = 0.0

        self.assertEqual(compute_f1(gold, pred), expected_f1)

    def test_intermediate_value(self):
        '''
        Tests if the F1 is intermediate for a case of intermediate match.
        '''
        pred = '1 2 3 4 5 6 7 8 9 10'
        gold = 'a 2 b 4 c 6 d 8 e 10'
        expected_f1 = 0.5

        self.assertEqual(compute_f1(gold, pred), expected_f1)
