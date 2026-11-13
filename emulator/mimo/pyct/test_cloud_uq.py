"""Deprecated: superseded by ``emulator/mimo/run_uq_analysis.py``.

The old script evaluated the random-tile CloudDataModule test split with an
aleatoric-only Laplace calibration curve and a precision-recall plot, and depended
on the external ``mimo`` package. All of that is now in ``run_uq_analysis.py``
(exact MIMO mixture predictive, Laplace or Gaussian, tables T1-T5, figures F1-F10).

To reproduce the old evaluation set, set ``include_test_split: true`` in
``emulator/mimo/config/uq_analysis.yaml``. This shim forwards to the new script::

    python test_cloud_uq.py -y ../config/uq_analysis.yaml [--stage all]
"""
import os
import runpy
import sys
import warnings

if __name__ == "__main__":
    warnings.warn("test_cloud_uq.py is deprecated; use emulator/mimo/run_uq_analysis.py",
                  DeprecationWarning, stacklevel=1)
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run_uq_analysis.py")
    sys.argv[0] = target
    runpy.run_path(target, run_name="__main__")
