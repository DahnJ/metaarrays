from typing import Any


def safeget(d: dict[Any, Any] | None, k: Any) -> Any | None:
    return d.get(k) if d else None
