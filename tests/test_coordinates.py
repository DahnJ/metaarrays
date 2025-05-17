import dask.array as da
import numpy as np
import xarray as xr

from metaarrays.coordinates import construct_metaarray_coordinates
from metaarrays.transform import Centroid, FirstCoordinate


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
            transforms={
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
            transforms={
                "time": FirstCoordinate(),
                "x": Centroid(),
                "y": Centroid(),
            },
        )

        assert actual.equals(expected)

    def test_coordinates_large(self) -> None:
        """Test performance with large datasets with 100 mil chunks."""
        ds = xr.Dataset(
            {
                "foo": (
                    ("time", "x", "y"),
                    da.empty((100, 10000, 10000), chunks=(1, 10, 10)),
                ),
                "bar": (
                    ("time", "x", "y"),
                    da.empty((100, 10000, 10000), chunks=(1, 10, 10)),
                ),
            },
            coords={
                "time": [
                    np.datetime64("2000-01") + np.timedelta64(i, "M")
                    for i in range(100)
                ],
                "x": 10 * np.arange(10000),
                "y": 10 * np.arange(10000),
            },
        )

        actual = construct_metaarray_coordinates(
            ds,
            transforms={
                "time": FirstCoordinate(),
                "x": Centroid(),
                "y": Centroid(),
            },
        )

        expected = xr.Dataset(
            coords={
                "time": [
                    np.datetime64("2000-01") + np.timedelta64(i, "M")
                    for i in range(0, 100)
                ],
                "x": 45 + np.arange(0, 10 * 10000, 100),
                "y": 45 + np.arange(0, 10 * 10000, 100),
            },
        )

        assert actual.equals(expected)
