import dask.array as da
import icechunk
import numpy as np
import xarray as xr
import zarr

from metaarrays.metaarray import construct_metaarray
from metaarrays.transform import Centroid, FirstCoordinate


class TestConstructMetaArray:
    def test_construct_metaarray(self):
        repo = icechunk.Repository.create(icechunk.in_memory_storage())
        group = "mydata"
        session = repo.writable_session("main")
        store = session.store
        ds = xr.Dataset(
            {
                "foo": (("x", "y"), da.empty((10, 10), chunks=(5, 5))),
                "bar": (("x", "y"), da.empty((10, 10), chunks=(5, 5))),
            },
            coords={
                "x": np.arange(10),
                "y": np.arange(10),
            },
        )
        ds.to_zarr(store=store, group=group, mode="w", compute=False)
        z = zarr.open(store, path=group, mode="a")
        z["foo"][0, 0] = 1

        actual = construct_metaarray(
            session=session,
            group=group,
            transforms={
                "x": Centroid(),
                "y": Centroid(),
            },
            variables=["foo", "bar"],
        )

        expected = xr.Dataset(
            {
                "foo": (("x", "y"), np.array([[1, 0], [0, 0]], dtype=np.int8)),
                "bar": (("x", "y"), np.array([[0, 0], [0, 0]], dtype=np.int8)),
            },
            coords={
                "x": np.array([2, 7]),
                "y": np.array([2, 7]),
            },
        )

        assert actual.equals(expected)

    def test_construct_metaarray_with_sel(self):
        repo = icechunk.Repository.create(icechunk.in_memory_storage())
        group = "mydata"
        session = repo.writable_session("main")
        store = session.store

        ds = xr.Dataset(
            {
                "foo": (("time", "x", "y"), da.empty((10, 10, 10), chunks=(1, 2, 2))),
                "bar": (("time", "x", "y"), da.empty((10, 10, 10), chunks=(1, 2, 2))),
            },
            coords={
                "time": [
                    np.datetime64("2010") + np.timedelta64(i, "Y") for i in range(10)
                ],
                "x": 10 * np.arange(10),
                "y": 10 * np.arange(10),
            },
        )
        transforms = {
            "time": FirstCoordinate(),
            "x": Centroid(),
            "y": Centroid(),
        }
        sel = {
            "time": [np.datetime64("2010"), np.datetime64("2011")],
            "x": slice(10, 30),
            "y": slice(30, 50),
        }
        ds.to_zarr(store=store, group=group, mode="w", compute=False)
        z = zarr.open(store, path=group, mode="a")
        z["foo"][0, 3, 3] = 1

        actual = construct_metaarray(
            session=session,
            group=group,
            transforms=transforms,
            variables=["foo", "bar"],
            sel=sel,
        )

        expected = xr.Dataset(
            {
                "foo": (
                    ("time", "x", "y"),
                    np.array([[[0, 0], [1, 0]], [[0, 0], [0, 0]]], dtype=np.int8),
                ),
                "bar": (
                    ("time", "x", "y"),
                    np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.int8),
                ),
            },
            coords={
                "time": [np.datetime64("2010", "ns"), np.datetime64("2011", "ns")],
                "x": np.array([5, 25]),
                "y": np.array([25, 45]),
            },
        )

        assert actual.equals(expected)

    def test_construct_metaarray_large(self):
        """Test performance with 10 bil chunks."""
        repo = icechunk.Repository.create(icechunk.in_memory_storage())
        group = "mydata"
        session = repo.writable_session("main")
        store = session.store
        n = 10000
        nt = 10000
        ds = xr.Dataset(
            {
                "foo": (
                    ("time", "x", "y"),
                    da.empty((nt, n, n), chunks=(nt, n, n)),
                ),
                "bar": (("time", "x", "y"), da.empty((nt, n, n), chunks=(nt, n, n))),
            },
            coords={
                "time": [
                    np.datetime64("2000-01") + np.timedelta64(i, "D") for i in range(nt)
                ],
                "x": 10 * np.arange(n),
                "y": 10 * np.arange(n),
            },
        )
        ds["foo"].encoding["chunks"] = (1, 10, 10)
        ds["bar"].encoding["chunks"] = (1, 10, 10)
        ds.to_zarr(store=store, group=group, mode="w", compute=False)
        z = zarr.open(store, path=group, mode="a")
        z["foo"][0, 11, 11] = 1

        transforms = {
            "time": FirstCoordinate(),
            "x": Centroid(),
            "y": Centroid(),
        }

        sel = {
            "time": [
                np.datetime64("2000-01-01", "ns"),
                np.datetime64("2000-01-02", "ns"),
            ],
            "x": slice(50, 200),
            "y": slice(50, 200),
        }

        actual = construct_metaarray(
            session=session,
            group=group,
            transforms=transforms,
            variables=["foo", "bar"],
            sel=sel,
        )

        expected = xr.Dataset(
            {
                "foo": (
                    ("time", "x", "y"),
                    np.array(
                        [
                            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        ],
                        dtype=np.int8,
                    ),
                ),
                "bar": (
                    ("time", "x", "y"),
                    np.array(
                        [
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        ],
                        dtype=np.int8,
                    ),
                ),
            },
            coords={
                "time": [
                    np.datetime64("2000-01-01", "ns"),
                    np.datetime64("2000-01-02", "ns"),
                ],
                "x": np.array([45, 145, 245]),
                "y": np.array([45, 145, 245]),
            },
        )

        assert actual.equals(expected)
