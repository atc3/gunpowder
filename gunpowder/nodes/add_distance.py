import logging

import numpy as np
import edt

from gunpowder.array import Array
from gunpowder.batch import Batch
from gunpowder.batch_request import BatchRequest

from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)


def _signed_distance(channel, anisotropy):
    """Signed distance of every voxel in ``channel`` (a binary 0/1 array) to
    the nearest label transition, positive inside the foreground.

    A direct EDT on ``channel`` would place the zero-crossing at whichever
    voxel happens to be background, biasing every distance by up to half a
    voxel. Instead, the label transitions are marked on a grid at half the
    voxel spacing (in between the original voxel centers, where the boundary
    actually lies), and the EDT of that finer grid is subsampled back down.
    Same technique as dacapo's ``DistancePredictor.process()``.
    """

    in_shape = channel.shape
    out_shape = tuple(2 * s - 1 for s in in_shape)
    boundaries = np.zeros(out_shape, dtype=bool)

    for d in range(len(in_shape)):
        shift_pos = [slice(None)] * len(in_shape)
        shift_pos[d] = slice(1, in_shape[d])
        shift_neg = [slice(None)] * len(in_shape)
        shift_neg[d] = slice(0, in_shape[d] - 1)

        target = [slice(None, None, 2)] * len(in_shape)
        target[d] = slice(1, out_shape[d], 2)

        boundaries[tuple(target)] = channel[tuple(shift_pos)] != channel[
            tuple(shift_neg)
        ]

    if not boundaries.any():
        # no label transition anywhere in this block: there is nothing to
        # take a distance to, so saturate towards the sign of whichever
        # label fills it, at the largest distance the block could support
        max_distance = min(s * v / 2 for s, v in zip(in_shape, anisotropy))
        sign = 1.0 if channel.flat[0] > 0 else -1.0
        return np.full(in_shape, sign * max_distance, dtype=np.float32)

    grid = (~boundaries).astype(np.uint8)
    distances = edt.edt(
        grid, anisotropy=tuple(v / 2 for v in anisotropy)
    ).astype(np.float32)
    distances = distances[(slice(None, None, 2),) * len(in_shape)]
    distances[channel == 0] *= -1

    return distances


class AddDistance(BatchFilter):
    """Add an array of signed distance transforms for a given label array.

    Args:

        labels (:class:`ArrayKey`):

            The array to read the labels from.

        distance (:class:`ArrayKey`):

            The array to generate containing the signed distance transform.

        scale_factor (float, optional):

            The amount by which to scale distances before applying a tanh normalization.

        distance_mask (:class:`ArrayKey`, optional):

            The array to generate containing the distance mask.

        dtype (np.dtype, optional):

            The data type to use for the distance array.
    """

    def __init__(
        self,
        labels,
        distance,
        scale_factor=20.0,
        distance_mask=None,
        dtype=np.float32,
    ):
        self.labels = labels
        self.distance = distance
        self.scale_factor = scale_factor
        self.distance_mask = distance_mask
        self.dtype = dtype

    def setup(self):
        assert self.labels in self.spec, (
            "Upstream does not provide %s needed by AddDistance" % self.labels
        )

        spec = self.spec[self.labels].copy()
        spec.dtype = self.dtype

        self.provides(self.distance, spec)
        if self.distance_mask:
            self.provides(self.distance_mask, spec)
        self.enable_autoskip()

    def prepare(self, request):
        deps = BatchRequest()

        deps[self.labels] = request[self.distance].copy()
        deps[self.labels].dtype = None
        deps[self.labels].roi = request[self.distance].roi

        return deps

    def process(self, batch, request):
        outputs = Batch()

        distance_roi = request[self.distance].roi

        logger.debug("computing signed distance transform from labels")

        logger.debug("labels shape %s", batch.arrays[self.labels].data.shape)

        voxel_size = self.spec[self.labels].voxel_size
        assert voxel_size is not None, (
            "AddDistance requires %s to have a voxel size" % self.labels
        )
        anisotropy = tuple(float(v) for v in voxel_size)

        n_dims = len(batch.arrays[self.labels].data.shape)
        if n_dims == voxel_size.dims:
            channel_dim = False
            # labels = labels[np.newaxis]
        elif n_dims == voxel_size.dims + 1:
            channel_dim = True
        else:
            raise NotImplementedError("Cannot handle multiple channel dims")

        labels = batch.arrays[self.labels].data.astype(np.int32)
        labels[labels > 0] = 1
        if not channel_dim:
            labels = labels[np.newaxis]

        distances = np.zeros(labels.shape, dtype=np.float32)
        for ii, channel in enumerate(labels):
            raw_distance = _signed_distance(channel, anisotropy)
            distances[ii] = np.tanh(raw_distance / self.scale_factor)

        logger.debug("distances shape %s", distances.shape)

        # crop affinities to requested ROI
        offset = distance_roi.offset
        crop_roi = distance_roi.shift(-offset)
        crop_roi /= voxel_size
        crop = crop_roi.get_bounding_box()

        logger.debug("cropping with " + str(crop))

        distances = distances[(slice(None),) + crop]

        spec = self.spec[self.distance].copy()
        spec.roi = distance_roi
        spec.dtype = self.dtype
        outputs.arrays[self.distance] = Array(distances.astype(self.dtype), spec)

        if self.distance_mask and self.distance_mask in request:

            mask = np.ones_like(distances)
            for channel_distance, channel_mask in zip(distances, mask):
                # pad every axis by 1 voxel to compute each voxel's distance
                # to the edge of this block -- except singleton axes (e.g.
                # the thickness-1 Z axis of a 2D patch requested as a 3D
                # ROI), which have no "interior" to be close to an edge of.
                # Padding them anyway would put a zero-plane exactly 1 voxel
                # from every single voxel in that axis, capping *every*
                # voxel's boundary distance at ~1 regardless of its true
                # in-plane distance to the block edge, and masking out the
                # entire block.
                pad = tuple(0 if s == 1 else 1 for s in channel_mask.shape)
                tmp = np.zeros(
                    tuple(s + 2 * p for s, p in zip(channel_mask.shape, pad)),
                    dtype=channel_mask.dtype,
                )
                slices = tuple(
                    slice(p, s + p) if p else slice(None)
                    for s, p in zip(channel_mask.shape, pad)
                )
                tmp[slices] = channel_mask
                boundary_distance = edt.edt(tmp, anisotropy=anisotropy)
                boundary_distance = np.tanh(
                    boundary_distance[slices] / self.scale_factor
                )

                logger.debug(
                    "Total number of masked in voxels before distance masking {0:}".format(
                        np.sum(channel_mask)
                    )
                )
                threshold = 0.8
                add = 5e-2
                # positive distance masking
                channel_mask[
                    np.logical_and(
                        np.clip(abs(channel_distance) + add, 0, threshold)
                        >= boundary_distance,
                        channel_distance >= 0,
                    )
                ] = 0
                logger.debug(
                    "Total number of masked in voxels after postive distance masking {0:}".format(
                        np.sum(channel_mask)
                    )
                )
                # negative distance masking
                channel_mask[
                    np.logical_and(
                        np.clip(abs(channel_distance) + add, 0, threshold)
                        >= boundary_distance,
                        channel_distance <= 0,
                    )
                ] = 0
                logger.debug(
                    "Total number of masked in voxels after negative distance masking {0:}".format(
                        np.sum(channel_mask)
                    )
                )

            # if no_channel_dim:
            #     mask = mask[0]

            mask = mask.astype(self.dtype)
            outputs.arrays[self.distance_mask] = Array(mask, spec)

        return outputs
