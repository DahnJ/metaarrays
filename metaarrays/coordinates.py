from typing import Hashable
import xarray as xr

from metaarrays.transformer import PixelToChunkLabelTransformer


def construct_metaarray_coordinates(
    ds: xr.Dataset, transformers: dict[Hashable, PixelToChunkLabelTransformer]
) -> xr.Dataset:
    coords = {}
    for name, transformer in transformers.items():
        coords[name] = transformer.transform(ds[name].data, ds.chunks[name])
    return xr.Dataset(
        {name: (ds[name].dims, coords[name]) for name in ds.coords},
    )
