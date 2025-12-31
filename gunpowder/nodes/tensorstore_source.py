import json
import logging
from collections.abc import MutableMapping
from typing import Union, List

import numpy as np

from gunpowder.ext import tensorstore as ts

from gunpowder.array import Array, ArrayKey
from gunpowder.array_spec import ArraySpec
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.roi import Roi

from gunpowder.nodes.batch_provider import BatchProvider

logger = logging.getLogger(__name__)


class TensorstoreSource(BatchProvider):
    """A `tensorstore <https://github.com/google/tensorstore/>`_ data source.

    Provides arrays from tensorstore datasets. If the attribute
    ``scale_metadata/resolution`` is set in a tensorstore dataset, it will be
    used as the array's ``scale_metadata/voxel_size``. If the attribute
    ``voxel_offset`` is set in a dataset, it will be used as the offset
    of the :class:`Roi` for this array. It is assumed that the offset is
    given in voxel units.

    Args:

        key (:class:`ArrayKey`):

            The ArrayKey for accessing this array.


        store (``string``, ``dict``):

            A dict representing a tensorstore config JSON,
            a JSON parseable string, or path to a tensorstore JSON config.

        array_spec (:class:`ArraySpec`, optional):

            An optional :class:`ArraySpec` to overwrite the array specs
            automatically determined from the tensorstore dataset. This is useful to set
            ``voxel_size``, for example. Only fields that are not ``None`` in
            the given :class:`ArraySpec` will be used.

        flatten_channels (``bool``, optional, default ``True``):

            If true and the dataset has a channels dimension of size 1,
            the channels dimension is removed. Default: ``True``.

        dim_order (``List[int]``, optional, default ``None``):

            The order in which to index data from the tensorstore dataset.
            If not provided, default to None = [0, 1, ..., ndims] (will index
            in the same order as the provided ROI).
    """

    def __init__(
        self,
        key: ArrayKey,
        store: Union[MutableMapping, str],
        array_spec=None,
        flatten_channels: bool = True,
        dim_order: List[int] = None,
    ):
        self.key = key

        # Store can either be a path to a JSON config,
        # a JSON parseable string, or a dict representing a tensorstore config JSON
        if isinstance(store, str):
            try:
                # Try to parse as JSON string
                logger.debug("Parsing TensorStore config from JSON string")
                store = json.loads(store)
            except json.JSONDecodeError:
                # Treat as path to JSON config
                logger.debug(
                    "Unable to parse as JSON string, loading TensorStore config from %s",
                    store,
                )
                store = json.load(open(store, "r"))
        elif isinstance(store, MutableMapping):
            logger.debug("Using TensorStore config from dict")
            pass

        self.store = store

        self.array_spec = array_spec
        self.flatten_channels = flatten_channels
        self.dim_order = dim_order

    def _open_file(self):
        ds = ts.open(self.store).result()
        if self.flatten_channels:
            logger.debug("Checking for singleton channel dimension to flatten")
            shape = ds.shape
            if len(shape) >= 1 and shape[-1] == 1:
                logger.debug("Found singleton channel dimension, flattening")
                ds = ds[ts.d["channel"][0]]
            else:
                logger.debug("No singleton channel dimension found")
        return ds

    def _get_voxel_size(self, data_file):
        try:
            dimension_units = data_file.schema.dimension_units
        except AttributeError:
            logger.debug("No dimension_units found in dataset schema")
            return None

        return Coordinate([x.multiplier for x in dimension_units])

    def _get_offset(self, data_file):
        try:
            domain = data_file.schema.domain
        except AttributeError:
            logger.debug("No domain found in dataset schema")
            return None

        return Coordinate(domain.inclusive_min)

    # Maybe use this in the future but unclear if tensorstore schema
    # stores the metadata to indicate if a file should be read in a different way
    # For now just default to [0, 1, 2, ...] and let the user decide for any
    # edge cases

    # def _get_dim_order(self, data_file):
    #     try:
    #         chunk_layout = data_file.schema.chunk_layout
    #     except AttributeError:
    #         logger.debug("No chunk_layout found in dataset schema")
    #         return None

    #     return list(chunk_layout.inner_order)

    def __read_spec(self, data_file):
        if self.array_spec is not None:
            spec = self.array_spec
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:
            voxel_size = self._get_voxel_size(data_file)
            if voxel_size is None:
                voxel_size = Coordinate((1,) * len(data_file.shape))
                logger.warning(
                    "WARNING: File %s does not contain resolution information "
                    "for %s, voxel size has been set to %s. This "
                    "might not be what you want.",
                    self.store,
                    self.key,
                    spec.voxel_size,
                )
            spec.voxel_size = voxel_size

        self.ndims = len(spec.voxel_size)

        if spec.roi is None:
            offset = self._get_offset(data_file)
            if offset is None:
                logger.debug("No offset found in dataset schema, setting to zero")
                offset = Coordinate((0,) * self.ndims)

            shape = Coordinate(data_file.shape[: self.ndims])
            spec.roi = Roi(offset, shape * spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == data_file.schema.dtype, (
                "dtype %s provided in array_specs for %s, "
                "but differs from dataset dtype %s"
                % (spec.dtype, self.key, data_file.schema.dtype)
            )
        else:
            spec.dtype = data_file.schema.dtype.numpy_dtype

        if spec.interpolatable is None:
            spec.interpolatable = np.issubdtype(spec.dtype, np.floating) or (
                spec.dtype == np.uint8
            )
            logger.warning(
                "WARNING: You didn't set 'interpolatable' for %s."
                "Based on the dtype %s, it has been "
                "set to %s. This might not be what you want.",
                self.key,
                spec.dtype,
                spec.interpolatable,
            )

        if self.dim_order is None:
            self.dim_order = list(range(self.ndims))
            # self.dim_order = self._get_dim_order(data_file)

        return spec

    def setup(self):
        data_file = self._open_file()
        spec = self.__read_spec(data_file)
        self.provides(self.key, spec)

    def provide(self, request):
        timing = Timing(self)
        timing.start()

        batch = Batch()

        data_file = self._open_file()

        logger.debug("Reading %s in %s...", self.key, request[self.key].roi)

        voxel_size = self.spec[self.key].voxel_size

        # scale request roi to voxel units
        dataset_roi = request[self.key].roi / voxel_size

        # shift request roi into dataset
        dataset_roi = dataset_roi - self.spec[self.key].roi.offset / voxel_size

        # create array spec
        array_spec = self.spec[self.key].copy()
        array_spec.roi = request[self.key].roi

        # add array to batch
        batch.arrays[self.key] = Array(
            self.__read(data_file, dataset_roi),
            array_spec,
        )

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read(self, data_file, roi):
        # number of channel dims
        c = len(data_file.shape) - self.ndims

        slices = roi.to_slices() + (slice(None),) * c
        # reorder slices according to dim_order
        slices = tuple([slices[i] for i in self.dim_order])

        array = np.asarray(data_file[slices].read().result())
        # If dim_order was not [0, 1, 2] or expected, then transpose by
        # the same to get back to a concordant shape with other datasets
        array = np.transpose(
            array, axes=[i + self.ndims for i in range(c)] + list(self.dim_order)
        )

        return array

    def name(self):
        return super().name() + f"[{json.dumps(self.store)}]"
