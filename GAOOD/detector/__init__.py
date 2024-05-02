# from .base import Detector
# from .base import DeepDetector
from .mybase import Detector
from .mybase import DeepDetector
from .GOOD_D import GOOD_D
from .GLocalKD import GLocalKD
from .GraphDE import GraphDE
from .GLADC import GLADC
from .SIGNET import SIGNET

__all__ = [
    "Detector", "DeepDetector", "GOOD_D","GraphDE","GLocalKD", "GLADC","SIGNET"
]
