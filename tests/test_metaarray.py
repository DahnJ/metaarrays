import dask.array as da
import icechunk
import numpy as np
import xarray as xr
import zarr

from metaarrays.metaarray import construct_metaarray
from metaarrays.transform import Centroid


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
            transform={
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
