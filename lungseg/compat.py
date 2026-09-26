"""Compatibility shims.

pylidc (last release 0.2.3) still uses ``np.int`` / ``np.float`` / ``np.bool``,
which were removed in NumPy 1.24, and imports ``pkg_resources``. Import this
module (or call :func:`import_pylidc`) instead of importing pylidc directly.
"""
import warnings

import numpy as np

for _name, _typ in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _typ)


def import_pylidc():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import pylidc as pl  # noqa: F401
        except ModuleNotFoundError as e:  # pragma: no cover
            if "pkg_resources" in str(e):
                raise ModuleNotFoundError(
                    "pylidc needs pkg_resources: pip install 'setuptools<81'") from e
            raise
    return pl
