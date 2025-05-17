from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

NumpyGenericT = TypeVar("NumpyGenericT", bound=np.generic | np.datetime64)
ndarray = NDArray[NumpyGenericT]
