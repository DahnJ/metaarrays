from abc import ABC, abstractmethod

import numpy as np

from numpy.typing import NDArray
from metaarrays.types import ndarray


class PixelToChunkLabelTransformer(ABC):
    @abstractmethod
    def transform(self, coordinates: ndarray, chunksizes: tuple[int, ...]) -> ndarray:
        """Translate pixel-level labels to chunk-level labels."""


class FirstCoordinate(PixelToChunkLabelTransformer):
    def transform(self, coordinates: ndarray, chunksizes: tuple[int, ...]) -> ndarray:
        chunk_bound_labels = _get_chunk_bound_labels(coordinates, chunksizes)
        return chunk_bound_labels[:, 0]


class Centroid(PixelToChunkLabelTransformer):
    def transform(self, coordinates: ndarray, chunksizes: tuple[int, ...]) -> ndarray:
        chunk_bound_labels = _get_chunk_bound_labels(coordinates, chunksizes)
        return np.mean(chunk_bound_labels, axis=1).astype(coordinates.dtype)  # type: ignore


def _get_chunk_bound_labels(coords: ndarray, chunksizes: tuple[int, ...]) -> ndarray:
    """Get the endpoints of chunks in a coordinate array.

    Example:
        >>> coords = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        >>> chunksizes = (3, 3, 2)
        >>> _get_coord_bounds(coords, chunksizes)
        array([[10, 30], [40, 60], [70, 80]])
    """
    if len(coords) != np.sum(chunksizes):
        raise ValueError("Length of coords must equal sum of chunks")

    chunk_bounds = _get_chunk_bound_indices(chunksizes)
    return coords[chunk_bounds]


def _get_chunk_bound_indices(chunksizes: tuple[int, ...]) -> NDArray[np.int_]:
    """Get the indices of chunks endpoints.

    Example:
        >>> chunksizes = (3, 3, 2)
        >>> _get_chunk_bounds(chunksizes)
        array([[0, 2], [3, 5], [6, 7]])
    """
    chunk_starts = np.concatenate([[0], np.cumsum(chunksizes)[:-1]])
    chunk_ends = np.cumsum(chunksizes) - 1
    return np.vstack([chunk_starts, chunk_ends]).T
