from typing import Any, Hashable

from metaarrays.bitmask import construct_bitmask
import numpy as np
import xarray as xr
from icechunk import Session
from numpy.typing import NDArray

from metaarrays.coordinates import construct_metaarray_coordinates
from metaarrays.icechunk import get_initialized_chunk_indices
from metaarrays.mapper import Mapper, construct_mappers
from metaarrays.transform import PixelToChunkLabelTransform


def construct_metaarray(
    session: Session,
    group: str,
    transforms: dict[Hashable, PixelToChunkLabelTransform],
    variables: list[str],
    sel: dict[Hashable, Any] | None = None,
) -> xr.Dataset:
    ds = xr.open_zarr(session.store, group=group, consolidated=False, zarr_version=3)
    mappers = construct_mappers(ds, transforms)
    coordinates = construct_metaarray_coordinates(ds, mappers, sel=sel)
    initialized = get_initialized_chunk_indices(session, group, variables)
    return _construct_dataset(coordinates, initialized, mappers, sel)


def _construct_dataset(
    coordinates: xr.Dataset,
    initialized: dict[Hashable, NDArray[np.int_]],
    mappers: dict[Hashable, Mapper],
    sel: dict[Hashable, Any] | None = None,
) -> xr.Dataset:
    coord_names = list(mappers.keys())
    data = {
        variable: (
            coord_names,
            construct_bitmask(
                coordinates,
                initialized_chunks,
                mappers,
                sel,
            ),
        )
        for variable, initialized_chunks in initialized.items()
    }

    return xr.Dataset(
        data,
        coords=coordinates.coords,
    )
