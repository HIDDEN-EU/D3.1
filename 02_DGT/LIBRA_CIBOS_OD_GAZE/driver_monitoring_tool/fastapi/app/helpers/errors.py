"""Domain exceptions raised by sessions and mapped to HTTP errors by the routes."""

class PointNotFoundError(LookupError):
    """The coordinates sent by the client match none of the session's reference points."""


class SessionStateError(Exception):
    """The request conflicts with the processor's current state (maps to 409)."""


class NoActiveSessionError(SessionStateError):
    """The request needs a session that hasn't been initialized."""

def point_not_found(target_point) -> PointNotFoundError:
    """Build a PointNotFoundError that lists the exact coordinates that would match."""
    x, y = (float(v) for v in target_point)
    return PointNotFoundError(f"No reference point at ({x!r}, {y!r})")