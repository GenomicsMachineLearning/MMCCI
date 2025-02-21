import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def align_dataframes(m1, m2, fill_value=0):
    """Aligns two DataFrames by matching their indices and columns, filling missing values with 0.

    Args:
        m1, m2 (pd.DataFrame): The DataFrames to align.

    Returns:
        tuple: A tuple of the aligned DataFrames.
    """

    m1, m2 = m1.align(m2, fill_value=fill_value)

    columns = sorted(set(m1.columns) | set(m2.columns))
    m1 = m1.reindex(columns, fill_value=fill_value)
    m2 = m2.reindex(columns, fill_value=fill_value)

    rows = sorted(set(m1.index) | set(m2.index))
    m1 = m1.reindex(rows, fill_value=fill_value)
    m2 = m2.reindex(rows, fill_value=fill_value)

    return m1, m2
