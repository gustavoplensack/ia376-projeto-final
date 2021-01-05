'''
Tests for implementations at src/metrics/utils.py
'''

import unittest

from src.metrics.utils import get_tokens, normalize_text


class TestNormalizeText(unittest.TestCase):
    def test_normalize_case(self):
        '''
        Test the sequence case normalization.
        '''
        sequence = 'I LiKe PiZzA'
        expected_sequence = 'i like pizza'

        self.assertEqual(normalize_text(sequence), expected_sequence)

    def test_space_normalization(self):
        '''
        Test the sequence spacing normalization.
        '''
        sequence = ' i    like   pizza '
        expected_sequence = 'i like pizza'

        self.assertEqual(normalize_text(sequence), expected_sequence)

    def test_space_and_case_normalization(self):
        '''
        Test the sequence spacing and case normalization.
        '''
        sequence = ' I    likE   PIZza '
        expected_sequence = 'i like pizza'

        self.assertEqual(normalize_text(sequence), expected_sequence)

    def test_already_normalized_sequence(self):
        '''
        Test if an already normalized sequence is not being changed as
        it passes through the function.
        '''
        sequence = 'i like pizza'
        expected_sequence = 'i like pizza'

        self.assertEqual(normalize_text(sequence), expected_sequence)


class TestGetTokens(unittest.TestCase):

    def test_empty_sequence(self):
        '''
        Test if the tokens for an empty sequence are tokenized as an empty
        list.
        '''
        sequence = ''
        expected_list = []

        self.assertEqual(get_tokens(sequence), expected_list)

    def test_regular_sequence(self):
        '''
        Test if a sequence is correctly tokenized.
        '''
        sequence = 'My name is Bond, James Bond.'
        expected_list = ['my', 'name', 'is', 'bond,', 'james', 'bond.']

        self.assertEqual(get_tokens(sequence), expected_list)

    def test_inavlid_sequence(self):
        '''
        Test the behavior for an non-str param to be tokenized.
        '''

        with self.assertRaises(ValueError):
            get_tokens(None)
