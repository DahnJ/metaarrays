from typing import Hashable

import xarray as xr

from metaarrays.transform import PixelToChunkLabelTransform


def construct_metaarray_coordinates(
    ds: xr.Dataset, transforms: dict[Hashable, PixelToChunkLabelTransform]
) -> xr.Dataset:
    coords = {}
    for name, transform in transforms.items():
        coords[name] = transform.transform(ds[name].values, ds.chunks[name])
    return xr.Dataset(
        {name: (ds[name].dims, coords[name]) for name in ds.coords},
    )
