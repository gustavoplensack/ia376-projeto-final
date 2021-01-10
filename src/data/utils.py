'''
This file implements utilites to handle data and datasets
'''
import glob
from typing import List


def load_text_jpeg_pairs(path_to_files: str) -> List[Tuple[str]]:
    '''
    Load text and jped samples from path_to_files

    Args:

        - path_to_files: The path to the files.

    Returns:

        - A list of tuples of [(path_to_image, path_to_text)].

    '''

    img_files = glob.glob(f'{path_to_files}/*.jpg')
    txt_files = glob.glob(f'{path_to_files}/*.txt')

    # sortind data, to assert that the data is organized
    img_files.sort()
    txt_files.sort()

    return [sample for sample in zip(img_files, txt_files)]
