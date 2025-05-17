import numpy as np
import asyncio

from collections.abc import Callable
from typing import Any, Iterable
from numpy.typing import NDArray
from icechunk import Session


def get_initialized_chunk_indices(
        session: Session,
        path: str,
        variables: Iterable[str],
    ) -> dict[str, NDArray[np.int_]]:
        async def _chunk_coordinates(path: str) -> NDArray[np.int_]:
            return np.array(
                list({c async for c in session.chunk_coordinates(path)})
            )

        async def _all_chunk_coordinates(
            variables: Iterable[str],
        ) -> dict[str, NDArray[np.int_]]:
            paths = [f"/{path}/{variable}" for variable in variables]
            tasks = [_chunk_coordinates(path) for path in paths]

            results = await asyncio.gather(*tasks)

            return dict(zip(variables, results))

        return _run_sync(
            _all_chunk_coordinates,
            variables=variables,
        )


def _run_sync(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run an async function in a synchronous context."""
    try:
        loop = asyncio.get_event_loop()
        loop_is_running = loop.is_running()
    except RuntimeError:
        loop_is_running = False

    if loop_is_running:
        return loop.run_until_complete(func(*args, **kwargs))
    return asyncio.run(func(*args, **kwargs))

