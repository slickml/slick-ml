from typing import Dict, Optional, Protocol, Union

import pandas as pd
from matplotlib.figure import Figure


class Metrics(Protocol):
    """Protocol for Metrics.

    Notes
    -----
    The main reason of this protocol is proper duck typing (PEP-544) [1]_ when using metrics such as
    ``RegressionMetrics`` or ``ClassificationMetrics`` in pipelines.

    References
    ----------
    .. [1] https://peps.python.org/pep-0544/
    """

    def plot(self) -> Figure:
        """Plots calculated metrics visualization.

        Returns
        -------
        Figure
        """
        ...  # pragma: no cover

    def get_metrics(
        self,
        dtype: Optional[str],
    ) -> Union[pd.DataFrame, Dict[str, Optional[float]]]:
        """Returns calculated metrics in a desired output dtype.

        Parameters
        ----------
        dtype : Optional[str]
            Metrics output dtype

        Returns
        -------
        Union[pd.DataFrame, Dict[str, Optional[float]]]
        """
        ...  # pragma: no cover
