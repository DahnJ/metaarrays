
# MetaArrays

This is a companion implementation for https://github.com/zarr-developers/zarr-specs/issues/300

This is not meant to be a usable package.

## What isn't included

For simplicity, the following features are not included:

- Any geospatial capability. We often query metaarrays using geospatial vector data, i.e. "what chunks are initialized in this polygon".
- Working with the result. We built functionality on top of the metaarray that enables dataframe-like operations on chunks to make it easier to work with the data. 