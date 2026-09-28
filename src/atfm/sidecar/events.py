"""Best-effort tool progress publication shared by live and completed output."""
from __future__ import annotations

from atfm.schema.events import ToolData, ToolProgress

from .parsers import default_parsers


def _safe_publish(bus, e) -> None:
    try:
        bus.publish(e)
    except Exception:
        pass


class ProgressEmitter:
    """Consume lines in parser order; publish each changed completed value once.

    A progress match stops the chain, including a duplicate. Data-only matches
    allow later parsers to run. Parser and publication failures never affect the
    tool's output or exit status.
    """

    def __init__(self, bus, session_id: str, call_id: str, parsers=None):
        self.bus, self.session_id, self.call_id = bus, session_id, call_id
        self.parsers = default_parsers() if parsers is None else parsers
        self.last_completed = None

    def feed(self, text: str, now: float) -> None:
        for p in self.parsers:
            try:
                prog = p.feed(text, now)
                if prog is not None:
                    if prog.get("completed") != self.last_completed:
                        self.last_completed = prog["completed"]
                        total = prog.get("total")
                        _safe_publish(self.bus, ToolProgress(
                            t=now, session_id=self.session_id, call_id=self.call_id,
                            completed=float(prog["completed"]),
                            total=None if total is None else float(total), phase=prog.get("phase"),
                        ))
                    break
                fd = getattr(p, "feed_data", None)
                if fd is not None:
                    d = fd(text, now)
                    if d is not None:
                        _safe_publish(self.bus, ToolData(
                            t=now, session_id=self.session_id, call_id=self.call_id,
                            metric=d["metric"], value=float(d["value"]),
                        ))
            except Exception:
                continue
