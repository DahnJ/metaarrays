from typing import Any, Hashable

import xarray as xr

from metaarrays.dict import safeget
from metaarrays.mapper import Level, Mapper, MapSpec, Space, Spec


def construct_metaarray_coordinates(
    ds: xr.Dataset,
    mappers: dict[Hashable, Mapper],
    sel: dict[Hashable, Any] | None = None,
) -> xr.Dataset:
    coords = {}
    spec = MapSpec(
        from_=Spec(level=Level.PIXEL, space=Space.LABEL),
        to=Spec(level=Level.CHUNK, space=Space.LABEL),
    )
    for name, mapper in mappers.items():
        coords[name] = mapper.map(spec, safeget(sel, name))

    return xr.Dataset(
        {name: (ds[name].dims, coords[name]) for name in ds.coords},
    )
