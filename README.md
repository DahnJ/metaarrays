
# MetaArrays

![](https://i.imgur.com/Ec65Iob.png)


## Motivation

There are two roughly two main ways to store earth observation data:

- Individual scenes in TIFF files
- Data cubes in Zarr

While data cubes bring many advantages, it is not always desirable to fully populate a data cube – instead, we may only want to populate the subset needed at any given time.

This is the pattern described in [Incrementally-populated Zarr Arrays](https://github.com/zarr-developers/zarr-specs/issues/300).

The natural question that arises with Zarr arrays that are not fully populated is:

> What areas of the data cube are populated?

Metaarays are a data structure that enables us to answer this question efficiently.

## How it works
For a code introduction, see [walkthrough](https://github.com/DahnJ/metaarrays/blob/main/walkthrough.ipynb).

![](https://i.imgur.com/dRs9VXK.png)

A MetaArray is just an Xarray DataArray where the elements represen the chunks themselves.

- The value `1` means the chunk has been populated. The value `0` means the chunk has not been populated.
- The coordinates are the centroids of the chunks. 

The second detail is important, as it allows us to use existing tooling to query the data.

It allows us to e.g. plot the chunk initialization information:

```python
data.plot()         # plot individual data pixels
metaarray.plot()    # plot the metaarray
```

![](https://i.imgur.com/9ChbPYc.png)

or query it geospatially using [rioxarray](https://github.com/corteva/rioxarray):


```python
metaarray.rio.clip([geometry], all_touched=True)
```


## Installation

Clone this repository and install the dependencies:

```bash
uv sync
```

## What isn't included

This repo is not a feature-full implementation, but rather an example implementation aimed at demonstrating the concept.

For simplicity, the following features are not included:

- Any geospatial capability. We often query metaarrays using geospatial vector data, i.e. "what chunks are initialized in this polygon".
- Querying interface. In our case, the user can point  to a zarr store and query by bbox/area/polygon/sel, without knowing anything about MetaArrays.
- Working with the query result. We built functionality on top of the metaarray that enables dataframe-like operations on chunks to make it easier to work with the result.