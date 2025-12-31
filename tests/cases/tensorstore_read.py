import numpy as np
import pytest

from gunpowder import (
    ArrayKey,
    ArraySpec,
    BatchRequest,
    Coordinate,
    MergeProvider,
    Roi,
    TensorstoreSource,
    build,
)
from gunpowder.ext import NoSuchModule, tensorstore as ts


@pytest.mark.skipif(isinstance(ts, NoSuchModule), reason="tensorstore is not installed")
def test_read(tmp_path):

    raw_key = ArrayKey("RAW")
    gt_key = ArrayKey("GT")

    roi_raw = Roi((20000, 2000, 2000), (2000, 400, 300))
    roi_gt = Roi((20100, 2010, 2010), (1800, 360, 270))
    voxel_size = Coordinate(20, 2, 2)

    # this generates 3-channel data -- mb support this in the future
    # raw_data = np.array(
    #     np.meshgrid(
    #         range((roi_raw / voxel_size).begin[0], (roi_raw / voxel_size).end[0]),
    #         range((roi_raw / voxel_size).begin[1], (roi_raw / voxel_size).end[1]),
    #         range((roi_raw / voxel_size).begin[2], (roi_raw / voxel_size).end[2]),
    #         indexing="ij",
    #     )
    # )
    # gt_data = np.array(
    #     np.meshgrid(
    #         range((roi_gt / voxel_size).begin[0], (roi_gt / voxel_size).end[0]),
    #         range((roi_gt / voxel_size).begin[1], (roi_gt / voxel_size).end[1]),
    #         range((roi_gt / voxel_size).begin[2], (roi_gt / voxel_size).end[2]),
    #         indexing="ij",
    #     )
    # )

    # 1-channel data to start
    ix, iy, iz = np.meshgrid(
        *[np.linspace(0, 1, n) for n in (roi_raw / voxel_size).shape[0:]], indexing="ij"
    )
    raw_data = (
        (np.abs(np.sin(4 * (ix + iy))) * 127) + (np.abs(np.sin(4 * (ix + iz))) * 127)
    ).astype(np.uint8)
    ix, iy, iz = np.meshgrid(
        *[np.linspace(0, 1, n) for n in (roi_gt / voxel_size).shape[0:]], indexing="ij"
    )
    gt_data = (
        (np.abs(np.sin(4 * (iy + iz))) * 127) + (np.abs(np.sin(4 * (ix + iy))) * 127)
    ).astype(np.uint16)

    raw_dataset = ts.open(
        {
            "driver": "n5",
            "dtype": "uint8",
            "kvstore": {
                "driver": "file",
                "path": str(tmp_path / "raw_dataset/"),
            },
            "metadata": {
                "compression": {"type": "gzip"},
                "dataType": "uint8",
                "dimensions": raw_data.shape,
                "blockSize": [64, 64, 64],
            },
            "create": True,
            "delete_existing": True,
        }
    ).result()
    gt_dataset = ts.open(
        {
            "driver": "n5",
            "dtype": "uint16",
            "kvstore": {
                "driver": "file",
                "path": str(tmp_path / "gt_dataset/"),
            },
            "metadata": {
                "compression": {"type": "gzip"},
                "dataType": "uint16",
                "dimensions": gt_data.shape,
                "blockSize": [64, 64, 64],
            },
            "create": True,
            "delete_existing": True,
        }
    ).result()

    raw_dataset[:] = raw_data
    gt_dataset[:] = gt_data

    pipeline = (
        TensorstoreSource(
            raw_key,
            {
                "driver": "n5",
                "kvstore": {
                    "driver": "file",
                    "path": str(tmp_path / "raw_dataset/"),
                },
            },
            array_spec=ArraySpec(roi_raw, voxel_size),
            dim_order=[0, 1, 2],
        ),
        TensorstoreSource(
            gt_key,
            {
                "driver": "n5",
                "kvstore": {
                    "driver": "file",
                    "path": str(tmp_path / "gt_dataset/"),
                },
            },
            array_spec=ArraySpec(roi_gt, voxel_size),
            dim_order=[0, 1, 2],
        ),
    ) + MergeProvider()
    request = BatchRequest()

    request[raw_key] = roi_raw
    request[gt_key] = roi_gt

    with build(pipeline):
        batch = pipeline.request_batch(request)

        assert (
            raw_data.shape[-3:]
            == batch[raw_key].spec.roi.shape // batch[raw_key].spec.voxel_size
        )
        assert roi_raw.offset == batch[raw_key].spec.roi.offset
        assert voxel_size == batch[raw_key].spec.voxel_size
        assert (raw_data == batch.arrays[raw_key].data).all()
