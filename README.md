
# MetaArrays
![](https://i.imgur.com/Ec65Iob.png)

This is an example implementation of MetaArrays a data structure used to 
track information about initialized chunks in an Icechunk Zarr store.

Start with the [walkthrough](https://github.com/DahnJ/metaarrays/blob/main/walkthrough.ipynb).

For more context, see https://github.com/zarr-developers/zarr-specs/issues/300.

This is not meant to be a usable package, although it could be made into one.

## Installation

Clone this repository and install the dependencies:

```bash
uv sync
```

## What isn't included

For simplicity, the following features are not included:

- Any geospatial capability. We often query metaarrays using geospatial vector data, i.e. "what chunks are initialized in this polygon".
- Querying interface. In our case, the user can point  to a zarr store and query by bbox/area/polygon/sel, without knowing anything about MetaArrays.
- Working with the query result. We built functionality on top of the metaarray that enables dataframe-like operations on chunks to make it easier to work with the result.