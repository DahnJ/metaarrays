from enum import IntEnum
from typing import Any, Hashable

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from metaarrays.dict import safeget
from metaarrays.mapper import Level, MapSpec, Mapper, Space, Spec


class ChunkState(IntEnum):
    UNINITIALIZED = 0
    INITIALIZED = 1


def construct_bitmask(
    coordinates: xr.Dataset,
    initialized: NDArray[np.int_],
    mappers: dict[Hashable, Mapper],
    sel: dict[Hashable, Any] | None = None,
) -> NDArray[np.int_]:
    min_, max_ = _find_min_max(mappers, sel)
    chunk_indices = _filter_chunk_indices(initialized, min_, max_)
    chunk_indices = _translate_chunk_indices(chunk_indices, min_)

    shape = tuple(coordinates[v].size for v in mappers.keys())
    bitmask = np.full(shape, ChunkState.UNINITIALIZED.value, dtype=np.int8)

    if len(chunk_indices) > 0:
        bitmask[tuple(chunk_indices.T)] = ChunkState.INITIALIZED.value

    return bitmask


def _find_min_max(
    mappers: dict[Hashable, Mapper],
    sel: dict[Hashable, Any] | None = None,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    spec = MapSpec(
        from_=Spec(level=Level.PIXEL, space=Space.LABEL),
        to=Spec(level=Level.CHUNK, space=Space.INDEX),
    )
    mins = []
    maxs = []
    for name in mappers.keys():
        chunk_index = mappers[name].map(spec, safeget(sel, name))
        mins.append(chunk_index.min())
        maxs.append(chunk_index.max())

    return np.array(mins), np.array(maxs)


def _filter_chunk_indices(
    chunk_indices: NDArray[np.int_],
    min_: NDArray[np.int_],
    max_: NDArray[np.int_],
) -> NDArray[np.int_]:
    if len(chunk_indices) == 0:
        return chunk_indices

    out_of_lower_bounds = chunk_indices < min_
    out_of_higher_bounds = chunk_indices > max_
    out_of_bounds = np.logical_or(out_of_lower_bounds, out_of_higher_bounds).any(axis=1)

    return chunk_indices[~out_of_bounds]


def _translate_chunk_indices(
    chunk_indices: NDArray[np.int_],
    min_: NDArray[np.int_],
) -> NDArray[np.int_]:
    if len(chunk_indices) == 0:
        return chunk_indices

    return chunk_indices - min_
