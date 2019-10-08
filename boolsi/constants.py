"""
Global BoolSi constants.
"""
from enum import Enum, IntEnum
try:
    from math import inf
except ImportError:
    from numpy import inf


class Mode(Enum):
    """
    BoolSi modes.
    """
    SIMULATE = 0
    ATTRACT = 1
    TARGET = 2


mode_descriptions = {Mode.SIMULATE: 'simulate', Mode.ATTRACT: 'attract', Mode.TARGET: 'target'}


class NodeStateRange(Enum):
    """
    Node state range codes.
    """
    MAYBE_FALSE = -0  # :)
    MAYBE_TRUE = -1
    MAYBE_TRUE_OR_FALSE = -10
    TRUE_OR_FALSE = 10


class MpiTags(IntEnum):
    """
    Tags for MPI.
    """
    READY = 0
    CONFIG = 1
    START = 2
    DONE = 3
    EXIT = 4
    ERROR = 5


simulation_name = 'simulation'
aggregated_attractor_name = 'attractor'

log_date_format = '%d-%b-%Y %H:%M:%S'
