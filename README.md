
# MetaArrays
![](https://private-user-images.githubusercontent.com/18722560/344239881-022e8fd4-6d92-4546-9b5e-b7a7ac00979d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDc0OTEyOTYsIm5iZiI6MTc0NzQ5MDk5NiwicGF0aCI6Ii8xODcyMjU2MC8zNDQyMzk4ODEtMDIyZThmZDQtNmQ5Mi00NTQ2LTliNWUtYjdhN2FjMDA5NzlkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MTclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTE3VDE0MDk1NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTc5MDJmOWFkN2ZiZTBlNjAzMDBiZTRmMDQ5NjU3MzJjMDY1OWZkNmQxODc5ZTY0MjUxZjA5NTA3OWI0MzhlN2UmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.5jtlW8WGnclfxrWhrNyg9p-illtIJ9K833zIpHJWhNc)

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