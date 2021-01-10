'''
Error raised when an experiment is misconfigurated
'''
from .base_error import Error


class ExperimentMisconfigurationError(Error):
    '''
    Error when a experiment has misconfigured params.
    '''

    def __init__(self, reason: str):
        super().__init__(f'Experiment is misconfigured: {reason}')
