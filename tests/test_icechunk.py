import numpy as np
import xarray as xr
import zarr
import dask.array as da

import icechunk
from metaarrays.icechunk import get_initialized_chunk_indices



class TestGetInitializedChunkIndices:
    def test_get_initilized_chunk_indices(self) -> None:
        repo = icechunk.Repository.create(icechunk.in_memory_storage())
        group = "mydata"
        session = repo.writable_session("main")
        store = session.store
        ds = xr.Dataset(
            {
                "foo": (("x", "y"), da.empty((10, 10), chunks=(2, 2))),
                "bar": (("x", "y"), da.empty((10, 10), chunks=(2, 2))),
            },
            coords={
                "x": np.arange(10),
                "y": np.arange(10),
            },
        )
        ds.to_zarr(store=store, group=group, mode="w", compute=False)
        z = zarr.open(store, path=group, mode='a')
        z["foo"][0, 0] = 1
        session.commit("write xarray data")


        session = repo.readonly_session("main")
        initialized = get_initialized_chunk_indices(
            session=session,
            path="mydata",
            variables=["foo", "bar"],
        )

        np.testing.assert_array_equal(
            initialized["foo"], 
            np.array([[0, 0]])
        )

        np.testing.assert_array_equal(
            initialized["bar"], 
            np.array([])
        )


