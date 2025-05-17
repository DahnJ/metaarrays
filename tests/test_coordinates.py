import numpy as np
import xarray as xr
import dask.array as da

from metaarrays.transformer import Centroid, FirstCoordinate
from metaarrays.coordinates import construct_metaarray_coordinates


class TestCoordinates:
    def test_coordinates(self) -> None:
        ds = xr.Dataset(
            {
                "foo": (("x", "y"), da.empty((10, 10), chunks=(2, 2))),
                "bar": (("x", "y"), da.empty((10, 10), chunks=(2, 2))),
            },
            coords={
                "x": 10 * np.arange(10),
                "y": 10 * np.arange(10),
            },
        )

        expected = xr.Dataset(
            coords={
                "x": np.array([5, 25, 45, 65, 85]),
                "y": np.array([5, 25, 45, 65, 85]),
            },
        )

        actual = construct_metaarray_coordinates(
            ds,
            transformers={
                "x": Centroid(),
                "y": Centroid(),
            },
        )

        assert actual.equals(expected)

    def test_coordinates_with_time(self) -> None:
        ds = xr.Dataset(
            {
                "foo": (("time", "x", "y"), da.empty((10, 10, 10), chunks=(2, 2, 2))),
                "bar": (("time", "x", "y"), da.empty((10, 10, 10), chunks=(2, 2, 2))),
            },
            coords={
                "time": [
                    np.datetime64("2010") + np.timedelta64(i, "Y") for i in range(10)
                ],
                "x": 10 * np.arange(10),
                "y": 10 * np.arange(10),
            },
        )

        expected = xr.Dataset(
            coords={
                "time": [
                    np.datetime64("2010") + np.timedelta64(i, "Y")
                    for i in range(0, 10, 2)
                ],
                "x": np.array([5, 25, 45, 65, 85]),
                "y": np.array([5, 25, 45, 65, 85]),
            },
        )

        actual = construct_metaarray_coordinates(
            ds,
            transformers={
                "time": FirstCoordinate(),
                "x": Centroid(),
                "y": Centroid(),
            },
        )

        assert actual.equals(expected)
