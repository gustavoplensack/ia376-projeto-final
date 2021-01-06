'''
This file implements utilites to handle data and datasets
'''
import glob
from typing import List


def load_text_jpeg_pairs(path_to_files: str) -> List[(str)]:
    '''
    Returns a list with tuples of [(path_to_image, path_to_text)].

    Args:
        - path_to_files (str): The path to the files.
    '''

    img_files = glob.glob(f'{path_to_files}/*.jpg')
    txt_files = glob.glob(f'{path_to_files}/*.txt')

    # sortind data, to assert that the data is organized
    img_files.sort()
    txt_files.sort()

    return [sample for sample in zip(img_files, txt_files)]
