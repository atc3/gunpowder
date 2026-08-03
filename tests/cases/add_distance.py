import numpy as np

from gunpowder import (
    AddDistance,
    Array,
    ArrayKey,
    ArraySpec,
    BatchRequest,
    Roi,
    build,
)

from .helper_sources import ArraySource

GT_LABELS = ArrayKey("GT_LABELS")
GT_DIST = ArrayKey("GT_DIST")
GT_DIST_MASK = ArrayKey("GT_DIST_MASK")

# raw distance (in world units) at which tanh(d / SCALE_FACTOR) == 0.8, i.e. the
# point at which the boundary mask starts protecting voxels whose distance value
# could be an underestimate due to the edge of the requested ROI.
SCALE_FACTOR = 20.0
MASK_THRESHOLD_DIST = SCALE_FACTOR * np.arctanh(0.8)  # ~= 21.97


def flat_boundary_source(shape, voxel_size, boundary):
    """A labels source that is 1 for axis-0 index < boundary, 0 otherwise."""

    data = np.zeros(shape, dtype=np.uint64)
    data[:boundary] = 1

    spec = ArraySpec(
        roi=Roi((0, 0, 0), tuple(s * v for s, v in zip(shape, voxel_size))),
        voxel_size=voxel_size,
        interpolatable=False,
    )
    return ArraySource(GT_LABELS, Array(data, spec))


def test_signed_distance_values():
    # flat boundary at z=5 (voxel index), spanning the whole y/x extent, so the
    # nearest background/foreground voxel to any interior point is straight
    # along z -- this makes the expected distance exactly computable even
    # though the voxel size is anisotropic.
    voxel_size = (20, 4, 8)
    shape = (11, 3, 3)
    boundary = 5

    pipeline = flat_boundary_source(shape, voxel_size, boundary) + AddDistance(
        GT_LABELS, GT_DIST
    )

    with build(pipeline):
        request = BatchRequest()
        roi = Roi((0, 0, 0), tuple(s * v for s, v in zip(shape, voxel_size)))
        request[GT_LABELS] = roi
        request[GT_DIST] = roi

        batch = pipeline.request_batch(request)

    dist = batch[GT_DIST].data
    assert dist.dtype == np.float32
    # AddDistance adds a leading channel axis when the input has none
    assert dist.shape == (1,) + shape

    # foreground occupies z indices [0, boundary), background [boundary, 11).
    # The label transition sits halfway between voxels boundary-1 and
    # boundary, not on a voxel center, so the raw (pre-tanh) distance of any
    # voxel z is (boundary - 0.5 - z) * voxel_size_z.
    y, x = 1, 1
    for z in [0, 4, 5, 10]:
        expected_raw = (boundary - 0.5 - z) * voxel_size[0]
        expected = np.tanh(expected_raw / SCALE_FACTOR)
        assert np.isclose(dist[0, z, y, x], expected, atol=1e-5), (
            f"z={z}: expected {expected}, got {dist[0, z, y, x]}"
        )


def test_anisotropy_changes_distance():
    # same flat boundary, but with isotropic voxels -- the raw distance to the
    # boundary is now in voxel units, not world units, so the result should
    # differ from the anisotropic case above. This guards against silently
    # dropping the anisotropy passed to edt.edt().
    shape = (11, 3, 3)
    boundary = 5

    pipeline = flat_boundary_source(shape, (1, 1, 1), boundary) + AddDistance(
        GT_LABELS, GT_DIST
    )

    with build(pipeline):
        request = BatchRequest()
        roi = Roi((0, 0, 0), shape)
        request[GT_LABELS] = roi
        request[GT_DIST] = roi

        batch = pipeline.request_batch(request)

    dist = batch[GT_DIST].data
    expected = np.tanh((boundary - 0.5) / SCALE_FACTOR)
    assert np.isclose(dist[0, 0, 1, 1], expected, atol=1e-5)
    # with voxel_size=20 in z, the same voxel gives tanh(90/20) instead
    assert not np.isclose(
        dist[0, 0, 1, 1], np.tanh((boundary - 0.5) * 20 / SCALE_FACTOR), atol=1e-3
    )


def test_uniform_block_saturates_without_inf():
    # a block with no label transition at all has no boundary to take a
    # distance to; edt.edt() on an all-foreground/all-background array
    # returns inf, so this has to be handled explicitly rather than by
    # relying on inf/tanh arithmetic happening to saturate to +-1.
    shape = (11, 3, 3)
    voxel_size = (20, 4, 8)

    data = np.ones(shape, dtype=np.uint64)
    roi = Roi((0, 0, 0), tuple(s * v for s, v in zip(shape, voxel_size)))
    spec = ArraySpec(roi=roi, voxel_size=voxel_size, interpolatable=False)
    pipeline = ArraySource(GT_LABELS, Array(data, spec)) + AddDistance(
        GT_LABELS, GT_DIST
    )

    with build(pipeline):
        request = BatchRequest()
        request[GT_LABELS] = roi
        request[GT_DIST] = roi

        batch = pipeline.request_batch(request)

    dist = batch[GT_DIST].data
    assert np.isfinite(dist).all()

    # largest distance the block can support: half the extent of its
    # shortest (in world units) axis
    max_distance = min(s * v / 2 for s, v in zip(shape, voxel_size))
    expected = np.tanh(max_distance / SCALE_FACTOR)
    assert np.allclose(dist, expected, atol=1e-5)


def test_dtype_is_applied_to_output():
    shape = (11, 3, 3)
    voxel_size = (20, 4, 8)

    pipeline = flat_boundary_source(shape, voxel_size, 5) + AddDistance(
        GT_LABELS, GT_DIST, dtype=np.float64
    )

    with build(pipeline):
        request = BatchRequest()
        roi = Roi((0, 0, 0), tuple(s * v for s, v in zip(shape, voxel_size)))
        request[GT_LABELS] = roi
        request[GT_DIST] = roi

        batch = pipeline.request_batch(request)

    assert batch[GT_DIST].data.dtype == np.float64
    assert batch[GT_DIST].spec.dtype == np.float64


def test_distance_mask_protects_roi_boundary():
    # an all-foreground volume has no internal label boundary at all, so the
    # only thing that can make a distance estimate unreliable is proximity to
    # the edge of the requested ROI -- exactly what distance_mask is meant to
    # flag.
    voxel_size = (20, 4, 8)
    shape = (11, 21, 21)

    data = np.ones(shape, dtype=np.uint64)
    roi = Roi((0, 0, 0), tuple(s * v for s, v in zip(shape, voxel_size)))
    spec = ArraySpec(roi=roi, voxel_size=voxel_size, interpolatable=False)
    source = ArraySource(GT_LABELS, Array(data, spec))

    pipeline = source + AddDistance(
        GT_LABELS, GT_DIST, distance_mask=GT_DIST_MASK
    )

    with build(pipeline):
        request = BatchRequest()
        request[GT_LABELS] = roi
        request[GT_DIST] = roi
        request[GT_DIST_MASK] = roi

        batch = pipeline.request_batch(request)

    mask = batch[GT_DIST_MASK].data
    assert mask.dtype == np.float32
    assert mask.shape == (1,) + shape

    # deep interior: far from every face of the ROI, so the distance estimate
    # (which saturates towards 1, since there is no background anywhere) is
    # trustworthy
    assert mask[0, 5, 10, 10] == 1

    # right at the z=0 / y=0 / x=0 faces the ROI could be hiding real
    # background just outside of what was requested, so those voxels are
    # masked out regardless of which axis is closest
    assert mask[0, 0, 10, 10] == 0
    assert mask[0, 5, 0, 10] == 0
    assert mask[0, 5, 10, 0] == 0


def test_distance_mask_handles_thickness_1_axis():
    # a 2D training patch requested as a 3D ROI with a thickness-1 Z axis
    # (the common case for a 2D model trained on slices sampled from a 3D
    # volume). Padding every axis by 1 voxel -- including a size-1 one --
    # to compute each voxel's distance to the block edge puts a zero-plane
    # exactly 1 voxel away from *every* voxel along that axis, capping every
    # voxel's boundary distance at ~1 regardless of true in-plane distance
    # to the Y/X edges, and masking out the entire block. Regression test
    # for that: a thickness-1 block should still have a trustworthy
    # (mask == 1) deep interior, just like a thick one does.
    #
    # isotropic voxel_size matters here: with the (20, 4, 8) anisotropy used
    # elsewhere in this file, being 1 voxel from the Z padding is already a
    # large *physical* distance (20), which happens not to trigger the bug --
    # this needs the isotropic (1, 1, 1) voxel size the real training
    # pipeline uses, where 1 voxel of Z padding is also physically tiny.
    voxel_size = (1, 1, 1)
    shape = (1, 21, 21)

    data = np.ones(shape, dtype=np.uint64)
    roi = Roi((0, 0, 0), tuple(s * v for s, v in zip(shape, voxel_size)))
    spec = ArraySpec(roi=roi, voxel_size=voxel_size, interpolatable=False)
    source = ArraySource(GT_LABELS, Array(data, spec))

    pipeline = source + AddDistance(
        GT_LABELS, GT_DIST, distance_mask=GT_DIST_MASK
    )

    with build(pipeline):
        request = BatchRequest()
        request[GT_LABELS] = roi
        request[GT_DIST] = roi
        request[GT_DIST_MASK] = roi

        batch = pipeline.request_batch(request)

    mask = batch[GT_DIST_MASK].data
    assert mask.shape == (1,) + shape

    # deep interior in Y/X: far from the Y/X faces, so despite the thickness-1
    # Z axis this should still be trustworthy, not masked out
    assert mask[0, 0, 10, 10] == 1

    # right at the y=0 / x=0 faces, still correctly masked regardless
    assert mask[0, 0, 0, 10] == 0
    assert mask[0, 0, 10, 0] == 0


def test_distance_mask_not_computed_when_not_requested():
    shape = (11, 3, 3)
    voxel_size = (20, 4, 8)

    pipeline = flat_boundary_source(shape, voxel_size, 5) + AddDistance(
        GT_LABELS, GT_DIST, distance_mask=GT_DIST_MASK
    )

    with build(pipeline):
        request = BatchRequest()
        roi = Roi((0, 0, 0), tuple(s * v for s, v in zip(shape, voxel_size)))
        request[GT_LABELS] = roi
        request[GT_DIST] = roi

        batch = pipeline.request_batch(request)

    assert GT_DIST in batch.arrays
    assert GT_DIST_MASK not in batch.arrays
