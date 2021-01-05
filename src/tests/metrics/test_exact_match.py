'''
This file implements tets for src/metrics/exact_match.py implementation.
'''

from unittest import TestCase

from src.metrics import compute_exact_match


class TestExactMatch(TestCase):

    def test_correct_match(self):
        '''
        Tests if the exact match is True for a correct match case.
        '''
        pred = 'My name is Bond, James Bond.'
        gold = 'My name is Bond, James Bond.'

        self.assertTrue(compute_exact_match(gold, pred))

    def test_incorrect_match(self):
        '''
        Tests if the exact match is False for an incorrect match case.
        '''
        pred = 'My name is Bond, James Bond.'
        gold = 'To the infinity and beyond!'

        self.assertFalse(compute_exact_match(gold, pred))
