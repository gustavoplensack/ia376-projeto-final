'''
This file implements the F1 metric calculation
'''

from collections import Counter

from .utils import get_tokens


def compute_f1(gold: str, pred: str) -> float:
    '''
    Computes F1 score with the formula below:

    2 * precision * recall / (precision + recall)


    Args:

        - gold: expected "gold" sequence.
        - pred: predicted sequence.

    Returns:

        - F1 score as a float [0,1]
    '''
    gold_toks = get_tokens(gold)
    pred_toks = get_tokens(pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1
