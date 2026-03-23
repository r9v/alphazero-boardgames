"""Diagnostic utilities for the training loop."""
import numpy as np


def raw_value_to_wdl_class(raw_v):
    """Convert raw values (+1/0/-1) to WDL class indices (0/1/2).

    Maps: +1 -> class 0 (win), 0 -> class 1 (draw), -1 -> class 2 (loss).
    """
    return (1 - raw_v).astype(np.int64)
