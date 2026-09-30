"""Validated wire contract for LMCache 0.5.5 multiprocess warm prefetch."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LMCacheConfig:
    url: str
    model_name: str
    chunk_size: int = 256
    world_size: int = 1
    cache_salt: str = ''
    timeout_s: float = 2.0
    completion_timeout_s: float = 2.0
    poll_interval_s: float = .02
    max_pending: int = 64
    wait_for_completion: bool = True   # False: submit and reconcile in release_expired (warms take seconds)

    def __post_init__(self):
        for value in (self.chunk_size, self.world_size, self.max_pending):
            if type(value) is not int or value < 1:
                raise ValueError('chunk_size, world_size and max_pending must be positive integers')
        for value in (self.timeout_s, self.completion_timeout_s, self.poll_interval_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('timeouts and polling interval must be positive and finite')
        if not self.model_name or not self.url.startswith(('http://', 'https://')):
            raise ValueError('model_name and an HTTP server URL are required')


def prefetch_body(cfg, tokens):
    if any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('token ids must be nonnegative integers')
    return dict(model_name=cfg.model_name, world_size=cfg.world_size, token_ids=list(tokens),
                cache_salt=cfg.cache_salt, source_tier='l2', target_tier='l1')


def submitted(response, chunks):
    if response.get('status') != 'submitted' or type(response.get('chunks')) is not int or response['chunks'] != chunks:
        raise ValueError('invalid prefetch acknowledgement')
    request_id = response.get('request_id')
    if not isinstance(request_id, str) or not request_id or not request_id.isalnum():
        raise ValueError('invalid request id')
    return request_id


def completed(response, request_id, expected):
    if response.get('request_id') != request_id:
        raise ValueError('prefetch request id mismatch')
    if response.get('status') == 'pending':
        return None
    found, total = response.get('found_keys'), response.get('total_keys')
    if (response.get('status') != 'completed' or type(found) is not int or type(total) is not int
            or total != expected or not 0 <= found <= total):
        raise ValueError('invalid prefetch completion')
    return dict(ok=found == total, status='completed' if found == total else 'partial',
                request_id=request_id, found_keys=found, total_keys=total, target='cpu_l1')
