"""Best-effort tool progress publication shared by live and completed output."""
from __future__ import annotations

from atfm.schema.events import ToolData, ToolProgress


def _safe_publish(bus, e) -> None:
    try:
        bus.publish(e)
    except Exception:
        pass


def publish_progress(text: str, now: float, parsers, bus, session_id: str, call_id: str, last_completed):
    """Return the latest completed value after best-effort parser publication.

    Progress matches (including duplicates) stop the chain; data matches continue.
    The caller owns parser state and chooses live or completion timestamps.
    """
    for p in parsers:
        try:
            prog = p.feed(text, now)
            if prog is not None:
                if prog.get("completed") != last_completed:
                    last_completed = prog["completed"]
                    total = prog.get("total")
                    _safe_publish(bus, ToolProgress(
                        t=now, session_id=session_id, call_id=call_id,
                        completed=float(prog["completed"]),
                        total=None if total is None else float(total), phase=prog.get("phase"),
                    ))
                break
            fd = getattr(p, "feed_data", None)
            if fd is not None:
                d = fd(text, now)
                if d is not None:
                    _safe_publish(bus, ToolData(
                        t=now, session_id=session_id, call_id=call_id,
                        metric=d["metric"], value=float(d["value"]),
                    ))
        except Exception:
            continue
    return last_completed
