"""Exact peer-rank counts; duplicate priorities share one histogram entry."""
from collections import Counter, defaultdict


def priority_quartile(below: int, total: int) -> int:
    return min(3, int((below / total) * 4)) if total else 3


class PeerRanks:
    def __init__(self):
        self.counts: dict[int, Counter[float]] = defaultdict(Counter)
        self.totals: Counter[int] = Counter()

    def add(self, entry):
        self.counts[entry.tier][entry.index] += 1
        self.totals[entry.tier] += 1

    def remove(self, entry):
        counts = self.counts[entry.tier]
        counts[entry.index] -= 1
        self.totals[entry.tier] -= 1
        if not counts[entry.index]:
            del counts[entry.index]
        if not counts:
            del self.counts[entry.tier]
            del self.totals[entry.tier]

    def bucket(self, tier, index):
        counts = self.counts.get(tier, {})
        below = sum(count for value, count in counts.items() if value < index)
        return priority_quartile(below, self.totals[tier])
