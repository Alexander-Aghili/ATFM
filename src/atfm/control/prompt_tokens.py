"""Resolve and cache session prompt tokens without assuming a backend layout."""
import hashlib
import json
from typing import Callable


class PromptTokens:
    """Session -> prefix token ids, cached until the session's prompt changes."""

    def __init__(self, tokenize: Callable[[list[dict]], list[int]], prompt_source: Callable[[str], list[dict] | None]):
        self.tokenize, self.prompt_source = tokenize, prompt_source
        self._cache: dict[str, tuple[str, list[int]]] = {}

    def get(self, session_id: str) -> list[int] | None:
        try:
            messages = self.prompt_source(session_id)
        except Exception:
            messages = None
        if not messages:
            return None
        key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        hit = self._cache.get(session_id)
        if hit is not None and hit[0] == key:
            return hit[1]
        toks = list(self.tokenize(messages))
        self._cache[session_id] = (key, toks)
        return toks
