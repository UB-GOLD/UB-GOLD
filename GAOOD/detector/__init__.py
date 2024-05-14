from .mybase import Detector
from .mybase import DeepDetector

from .GOOD_D import GOOD_D
from .GLocalKD import GLocalKD
from .GraphDE import GraphDE
from .GLADC import GLADC
from .SIGNET import SIGNET
from .CVTGAD import CVTGAD
from .OCGTL import OCGTL
from .OCGIN import OCGIN

__all__ = [
    "Detector", "DeepDetector", "GOOD_D","GraphDE","GLocalKD", "GLADC","SIGNET", "CVTGAD", "OCGTL","OCGIN"
]
