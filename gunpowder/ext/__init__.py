from __future__ import print_function

import logging
import sys
import traceback
from typing import Any, Optional

logger = logging.getLogger(__name__)


class NoSuchModule(object):
    def __init__(self, name):
        self.__name = name
        self.__traceback_str = traceback.format_tb(sys.exc_info()[2])
        errtype, value = sys.exc_info()[:2]
        assert errtype is not None
        self.__exception = errtype(value)

    def __getattr__(self, item):
        raise self.__exception


try:
    import dvision  # ty: ignore[import-not-found]
except ImportError:
    dvision = NoSuchModule("dvision")

try:
    import h5py
except ImportError:
    h5py = NoSuchModule("h5py")

try:
    import pyklb  # ty: ignore[import-not-found]
except ImportError:
    pyklb = NoSuchModule("pyklb")

try:
    import tensorflow  # ty: ignore[import-not-found]
except ImportError:
    tensorflow = NoSuchModule("tensorflow")

try:
    import torch  # ty: ignore[import-not-found]
except ImportError:
    torch = NoSuchModule("torch")

try:
    import tensorboardX  # ty: ignore[import-not-found]
except ImportError:
    tensorboardX = NoSuchModule("tensorboardX")

try:
    import malis  # ty: ignore[import-not-found]
except ImportError:
    malis = NoSuchModule("malis")

try:
    import augment
except ImportError:
    augment = NoSuchModule("augment")

ZarrFile: Optional[Any] = None
try:
    import zarr

    from .zarr_file import ZarrFile
except ImportError:
    zarr = NoSuchModule("zarr")
    ZarrFile = None

try:
    import daisy  # ty: ignore[import-not-found]
except ImportError:
    daisy = NoSuchModule("daisy")

try:
    import jax  # ty: ignore[import-not-found]
except ImportError:
    jax = NoSuchModule("jax")

try:
    import jax.numpy as jnp  # ty: ignore[import-not-found]
except ImportError:
    jnp = NoSuchModule("jnp")

try:
    import haiku  # ty: ignore[import-not-found]
except ImportError:
    haiku = NoSuchModule("haiku")

try:
    import optax  # ty: ignore[import-not-found]
except ImportError:
    optax = NoSuchModule("optax")

try:
    import tensorstore  # ty: ignore[import-not-found]
except ImportError:
    tensorstore = NoSuchModule("tensorstore")
