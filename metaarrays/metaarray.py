from typing import Hashable
from icechunk import Session
import numpy as np
from numpy.typing import NDArray
import xarray as xr

from metaarrays.coordinates import construct_metaarray_coordinates
from metaarrays.transformer import PixelToChunkLabelTransformer
from metaarrays.icechunk import get_initialized_chunk_indices


from enum import IntEnum


class ChunkState(IntEnum):
    UNINITIALIZED = 0
    INITIALIZED = 1


def construct_metaarray(
    session: Session,
    group: str,
    transformers: dict[Hashable, PixelToChunkLabelTransformer],
    variables: list[str],
) -> xr.Dataset:
    ds = xr.open_zarr(session.store, group=group, consolidated=False, zarr_version=3)
    coordinates = construct_metaarray_coordinates(ds, transformers)
    initialized = get_initialized_chunk_indices(session, group, variables)
    return _contruct_dataset(coordinates, initialized)


def _contruct_dataset(
    coordinates: xr.Dataset,
    initialized: dict[Hashable, NDArray[np.int_]],
) -> xr.Dataset:
    coord_names = list(coordinates.coords)
    data = {
        variable: (coord_names, _construct_values(coordinates, initialized_chunks))
        for variable, initialized_chunks in initialized.items()
    }

    return xr.Dataset(
        data,
        coords=coordinates.coords,
    )


def _construct_values(
    coordinates: xr.Dataset,
    initialized: NDArray[np.int_],
) -> NDArray[np.int_]:
    shape = tuple(coordinates[v].size for v in coordinates.coords)
    bitmask = np.full(shape, ChunkState.UNINITIALIZED.value, dtype=np.int8)

    if len(initialized) > 0:
        bitmask[tuple(initialized.T)] = ChunkState.INITIALIZED.value

    return bitmask
