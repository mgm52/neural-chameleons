from typing import List, Union

import numpy as np
from jaxtyping import Float
from matplotlib.figure import Figure
from torch import Tensor, nn

Params = Union[nn.ParameterList, str]
LoggingData = Union[np.ndarray, Tensor, int, Float, float, str, Figure]
