'''
Utility functions to normalize text and split into tokens.
'''

from typing import List


def normalize_text(s: str) -> str:
    """
        Lower text and remove extra whitespaces.

        Args:

            - s: string to be normalized.

        Returns:

            - text with lower case and removed extra whitespaces.

    """

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(lower(s))


def get_tokens(s: str) -> List[str]:
    '''
    Turn a string into a list of tokens, when the values are separed by commas.
    The sequence is normalized after being processed by the model.

    Args:

        - s: target string.

    Returns:

        - A list with the normalized words as tokens (separed by a whitespace).
    '''
    if not isinstance(s, str):
        raise ValueError('s must be a string object!')
    return normalize_text(s).split()
