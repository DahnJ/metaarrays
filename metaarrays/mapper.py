from enum import Enum
from typing import Any, Sequence

import numpy as np
import xarray as xr
from pydantic import BaseModel

from metaarrays.transform import PixelToChunkLabelTransform
from metaarrays.types import ndarray


class Space(Enum):
    INDEX = 0
    LABEL = 1


class Level(Enum):
    PIXEL = 0
    CHUNK = 1


class Spec(BaseModel):
    level: Level
    space: Space


class MapSpec(BaseModel):
    from_: Spec
    to: Spec


class Mapper(BaseModel):
    coordinate: xr.DataArray
    chunksizes: tuple[int, ...] | None = None
    transform: PixelToChunkLabelTransform | None = None

    class Config:
        arbitrary_types_allowed = True

    def map(self, specification: MapSpec, value: Any | None = None) -> ndarray:
        if self.chunksizes is None and (
            specification.from_.level == Level.CHUNK
            or specification.to.level == Level.CHUNK
        ):
            raise ValueError("Cannot map at chunk level without chunksizes.")

        if self.transform is None and specification.to == Spec(
            level=Level.CHUNK, space=Space.LABEL
        ):
            raise ValueError("Cannot map to chunk label without a transform. ")

        pixel_index: Any
        match specification.from_:
            case Spec(level=Level.PIXEL, space=Space.LABEL):
                pixel_index = _map_pixel_label_to_pixel_index(self.coordinate, value)
            case Spec(level=Level.PIXEL, space=Space.INDEX):
                pixel_index = _map_pixel_index_to_pixel_index(self.coordinate, value)
            case Spec(level=Level.CHUNK, space=Space.INDEX):
                pixel_index = _map_chunk_index_to_pixel_index(
                    self.coordinate, self.chunksizes, value
                )
            case Spec(level=Level.CHUNK, space=Space.LABEL):
                raise NotImplementedError(
                    "Mapping from chunk label to pixel index is not implemented."
                    "Use pixel label instead."
                )

        match specification.to:
            case Spec(level=Level.PIXEL, space=Space.LABEL):
                return _map_pixel_index_to_pixel_label(self.coordinate, pixel_index)
            case Spec(level=Level.PIXEL, space=Space.INDEX):
                return pixel_index
            case Spec(level=Level.CHUNK, space=Space.INDEX):
                return _map_pixel_index_to_chunk_index(
                    self.coordinate, self.chunksizes, pixel_index
                )
            case Spec(level=Level.CHUNK, space=Space.LABEL):
                return _map_pixel_index_to_chunk_label(
                    self.coordinate,
                    self.chunksizes,
                    pixel_index,
                    self.transform,
                )


def _map_pixel_index_to_pixel_index(
    coords: xr.DataArray, sel: Any | None
) -> np.ndarray:
    map_ = xr.DataArray(
        np.arange(len(coords)),
        dims=[coords.name],
        coords={coords.name: np.arange(len(coords))},
    )
    if sel is None:
        return map_.values
    return map_.sel({coords.name: sel}).values


def _map_pixel_label_to_pixel_index(coord: xr.DataArray, sel: Any | None) -> np.ndarray:
    map_ = xr.DataArray(
        data=np.arange(len(coord)),
        coords={coord.name: coord},
    )
    if sel is None:
        return map_.values
    return map_.sel({coord.name: sel}).values


def _map_pixel_index_to_pixel_label(
    coords: xr.DataArray, sel: Any | None
) -> np.ndarray:
    if sel is None:
        return coords.values
    return coords.isel({coords.name: sel}).values


def _map_pixel_index_to_chunk_index(
    coords: xr.DataArray, chunksizes: Sequence[int], sel: Any | None
) -> np.ndarray:
    map_ = xr.DataArray(
        np.repeat(np.arange(len(chunksizes)), chunksizes),
        dims=[coords.name],
        coords={coords.name: np.arange(len(coords))},
    )
    if sel is None:
        return np.unique(map_.values)
    return np.unique(map_.isel({coords.name: sel}).values)


def _map_chunk_index_to_pixel_index(
    coords: xr.DataArray, chunksizes: Sequence[int], sel: Any | None
) -> np.ndarray:
    map_ = xr.DataArray(
        np.arange(len(coords)),
        dims=[coords.name],
        coords={coords.name: np.repeat(np.arange(len(chunksizes)), chunksizes)},
    )
    if sel is None:
        return map_.values
    if isinstance(sel, list):
        raise NotImplementedError(
            "Mapping from chunk index with a list of indices is not implemented."
        )
    return map_.sel({coords.name: sel}).values


def _map_pixel_index_to_chunk_label(
    coords: xr.DataArray,
    chunksizes: Sequence[int],
    sel: Any | None,
    transform: PixelToChunkLabelTransform,
) -> np.ndarray:
    chunk_index = _map_pixel_index_to_chunk_index(coords, chunksizes, sel)
    chunk_labels = transform.transform(coords.values, chunksizes)
    map_ = xr.DataArray(
        chunk_labels,
        dims=[coords.name],
        coords={coords.name: np.arange(len(chunk_labels))},
    )
    return map_.sel({coords.name: chunk_index}).values
