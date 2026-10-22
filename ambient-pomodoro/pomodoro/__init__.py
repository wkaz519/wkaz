"""Ambient Pomodoro: a focus timer over slowly evolving generative backgrounds."""

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# The noise code multiplies many *small* matrices; multi-threaded BLAS only
# adds overhead there (measured ~6x slower) and would keep every core busy.
# Must be set before NumPy is first imported.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

__version__ = "0.1.0"
