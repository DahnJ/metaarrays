from typing import Any

import numpy as np
import pytest
import xarray as xr

from metaarrays.mapper import (
    Level,
    Mapper,
    MapSpec,
    Space,
    Spec,
)
from metaarrays.transform import Centroid, FirstCoordinate

(Centroid,)


class TestMapper:
    @pytest.mark.parametrize(
        ("spec", "value", "expected"),
        [
            (
                MapSpec(
                    from_=Spec(level=Level.PIXEL, space=Space.INDEX),
                    to=Spec(level=Level.PIXEL, space=Space.INDEX),
                ),
                slice(3, 10),
                np.array([3, 4, 5, 6, 7, 8, 9]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.PIXEL, space=Space.INDEX),
                    to=Spec(level=Level.PIXEL, space=Space.INDEX),
                ),
                3,
                np.array(3),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.PIXEL, space=Space.INDEX),
                    to=Spec(level=Level.PIXEL, space=Space.INDEX),
                ),
                [3],
                np.array([3]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.PIXEL, space=Space.LABEL),
                    to=Spec(level=Level.PIXEL, space=Space.INDEX),
                ),
                slice(30, 100),
                np.array([3, 4, 5, 6, 7, 8, 9]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.PIXEL, space=Space.LABEL),
                    to=Spec(level=Level.PIXEL, space=Space.INDEX),
                ),
                [30, 40],
                np.array([3, 4]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.PIXEL, space=Space.LABEL),
                    to=Spec(level=Level.PIXEL, space=Space.LABEL),
                ),
                slice(30, 40),
                np.array([30, 40]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.PIXEL, space=Space.LABEL),
                    to=Spec(level=Level.CHUNK, space=Space.INDEX),
                ),
                slice(30, 40),
                np.array([1, 2]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.PIXEL, space=Space.INDEX),
                    to=Spec(level=Level.CHUNK, space=Space.INDEX),
                ),
                [2, 8],
                np.array([1, 4]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.CHUNK, space=Space.INDEX),
                    to=Spec(level=Level.PIXEL, space=Space.INDEX),
                ),
                3,
                np.array([6, 7]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.CHUNK, space=Space.INDEX),
                    to=Spec(level=Level.PIXEL, space=Space.INDEX),
                ),
                slice(1, 3),
                np.array([2, 3, 4, 5, 6, 7]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.CHUNK, space=Space.INDEX),
                    to=Spec(level=Level.CHUNK, space=Space.INDEX),
                ),
                slice(1, 3),
                np.array([1, 2, 3]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.PIXEL, space=Space.LABEL),
                    to=Spec(level=Level.CHUNK, space=Space.INDEX),
                ),
                None,
                np.array([0, 1, 2, 3, 4]),
            ),
        ],
    )
    def test_mapper(
        self,
        spec: MapSpec,
        value: Any,
        expected: np.ndarray,
    ) -> None:
        coordinate = xr.Dataset(coords={"x": 10 * np.arange(10)})["x"]
        chunksizes = (2,) * 5

        mapper = Mapper(coordinate=coordinate, chunksizes=chunksizes)

        result = mapper.map(
            specification=spec,
            value=value,
        )

        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        ("spec", "transform", "value", "expected"),
        [
            (
                MapSpec(
                    from_=Spec(level=Level.PIXEL, space=Space.INDEX),
                    to=Spec(level=Level.CHUNK, space=Space.LABEL),
                ),
                Centroid(),
                slice(30, 40),  # out of bounds
                np.array([]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.PIXEL, space=Space.INDEX),
                    to=Spec(level=Level.CHUNK, space=Space.LABEL),
                ),
                Centroid(),
                slice(3, 4),
                np.array([25, 45]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.PIXEL, space=Space.LABEL),
                    to=Spec(level=Level.CHUNK, space=Space.LABEL),
                ),
                FirstCoordinate(),
                slice(30, 40),
                np.array([20, 40]),
            ),
            (
                MapSpec(
                    from_=Spec(level=Level.CHUNK, space=Space.INDEX),
                    to=Spec(level=Level.CHUNK, space=Space.LABEL),
                ),
                FirstCoordinate(),
                slice(1, 2),
                np.array([20, 40]),
            ),
        ],
    )
    def test_mapper_with_transform(
        self,
        spec: MapSpec,
        transform: Any,
        value: Any,
        expected: np.ndarray,
    ) -> None:
        coordinate = xr.Dataset(coords={"x": 10 * np.arange(10)})["x"]
        chunksizes = (2,) * 5

        mapper = Mapper(
            coordinate=coordinate, chunksizes=chunksizes, transform=transform
        )

        result = mapper.map(
            specification=spec,
            value=value,
        )

        assert np.array_equal(result, expected)
