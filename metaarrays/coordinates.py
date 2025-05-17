from typing import Any, Hashable

import xarray as xr

from metaarrays.dict import safeget
from metaarrays.mapper import Level, Mapper, MapSpec, Space, Spec
from metaarrays.transform import PixelToChunkLabelTransform


def construct_metaarray_coordinates(
    ds: xr.Dataset,
    transforms: dict[Hashable, PixelToChunkLabelTransform],
    sel: dict[Hashable, Any] | None = None,
) -> xr.Dataset:
    coords = {}
    spec = MapSpec(
        from_=Spec(level=Level.PIXEL, space=Space.LABEL),
        to=Spec(level=Level.CHUNK, space=Space.LABEL),
    )
    # TODO: Refactor to mapper?
    for name, transform in transforms.items():
        mapper = Mapper(
            coordinate=ds[name], chunksizes=ds.chunksizes[name], transform=transform
        )
        coords[name] = mapper.map(spec, safeget(sel, name))

    return xr.Dataset(
        {name: (ds[name].dims, coords[name]) for name in ds.coords},
    )
