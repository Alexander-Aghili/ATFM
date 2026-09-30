"""LMCache MP 0.5.5: secondary-storage to CPU warming, with verified completion.

This API holds no pin leases. GPU placement, pin, unpin and arbitrary move are
unsupported; the previous legacy adapter's assumed endpoints are not used.
"""
import time

from .lmcache_protocol import LMCacheConfig, completed, prefetch_body, submitted
from .prompt_tokens import PromptTokens


class LMCacheActuator:
    def __init__(self, cfg: LMCacheConfig, client=None, tokens: PromptTokens | None = None):
        import httpx
        self.cfg, self.tokens = cfg, tokens
        self._owns_client = client is None
        self.client = client if client is not None else httpx.Client(timeout=cfg.timeout_s)
        self.pending = {}
        self._verified = False
        self.errors = 0
        self.outcomes = {}

    def close(self):
        if self._owns_client:
            self.client.close()

    def _request(self, method, path, status, **kwargs):
        response = self.client.request(method, self.cfg.url.rstrip('/') + path, **kwargs)
        if response.status_code == 404:
            raise LookupError('prefetch status unavailable; completion is unknown')
        if response.status_code != status:
            raise ValueError(f'LMCache returned HTTP {response.status_code}')
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError('LMCache returned a non-object response')
        return result

    def prefetch(self, session_id):
        try:
            tokens = tuple(self.tokens.get(session_id) or ()) if self.tokens else ()
            if not tokens:
                return dict(ok=False, status='no_prompt')
            self.verify_backend()
            body = prefetch_body(self.cfg, tokens)
            chunks = len(tokens) // self.cfg.chunk_size
            if chunks == 0:
                return dict(ok=False, status='no_complete_chunk')
            if session_id not in self.pending:
                self._submit(session_id, tokens, body, chunks)
            return self._follow(session_id, tokens)
        except Exception as exc:
            self.errors += 1
            return dict(ok=False, status='error', reason=str(exc))

    def _follow(self, session_id, tokens):
        original, request_id, expected = self.pending[session_id]
        if original != tokens:
            return dict(ok=False, status='previous_prompt_pending', request_id=request_id)
        if not self.cfg.wait_for_completion:
            return dict(ok=None, status='submitted', request_id=request_id)
        return self._wait(session_id, request_id, expected)

    def verify_backend(self):
        if self._verified:
            return
        response = self.client.request('GET', self.cfg.url.rstrip('/') + '/version')
        version = response.json()
        if response.status_code != 200 or not isinstance(version, str) or version.split('-')[0] != '0.5.5':
            raise ValueError('requires the validated LMCache MP 0.5.5 API')
        status = self._request('GET', '/status', 200)
        if status.get('chunk_size') != self.cfg.chunk_size or status.get('is_healthy') is not True:
            raise ValueError('LMCache chunk size mismatch or unhealthy backend')
        self._verified = True

    def _submit(self, session_id, tokens, body, chunks):
        if len(self.pending) >= self.cfg.max_pending:
            raise ValueError('prefetch pending limit reached')
        result = self._request('POST', '/cache/prefetches', 202, json=body)
        request_id = submitted(result, chunks)
        self.pending[session_id] = (tokens, request_id, chunks * self.cfg.world_size)

    def _poll(self, session_id, request_id, expected):
        try:
            response = self._request('GET', '/cache/prefetches/' + request_id, 200)
        except LookupError:
            del self.pending[session_id]
            return self._settled(dict(ok=False, status='unknown', request_id=request_id))
        result = completed(response, request_id, expected)
        if result is not None:
            del self.pending[session_id]
            self._settled(result)
        return result

    def _settled(self, result):
        self.outcomes[result['status']] = self.outcomes.get(result['status'], 0) + 1
        return result

    def _wait(self, session_id, request_id, expected):
        deadline = time.monotonic() + self.cfg.completion_timeout_s
        while True:
            result = self._poll(session_id, request_id, expected)
            if result is not None:
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return dict(ok=False, status='pending', request_id=request_id)
            time.sleep(min(self.cfg.poll_interval_s, remaining))

    def release_expired(self, now):
        """Reconcile pending warm jobs; MP warming holds no leases to release."""
        for session_id, (_, request_id, expected) in list(self.pending.items()):
            try:
                self._poll(session_id, request_id, expected)
            except Exception:
                self.errors += 1
        return []

    def apply_touch(self, directive, now):
        return dict(ok=False, status='expired' if directive.expired(now) else 'unsupported', operation='pin')

    def apply_tier(self, directive, now):
        if directive.expired(now):
            return dict(ok=False, status='expired')
        if directive.action != 'prefetch' or directive.tier not in ('cpu', 'l1'):
            return dict(ok=False, status='unsupported', operation=directive.action, tier=directive.tier)
        return self.prefetch(directive.session_id)
