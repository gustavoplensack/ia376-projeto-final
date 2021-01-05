'''
This file implements the exact match metric.
'''

from .utils import normalize_text


def compute_exact_match(gold: str, pred: str) -> bool:
    '''
    Computes the exact match for normalized_text

    Args:

        - gold: expected "gold" sequence.
        - pred: predicted sequence.

    Returns:

        - Boolean if the match is exact.
    '''
    return int(normalize_text(gold) == normalize_text(pred))
