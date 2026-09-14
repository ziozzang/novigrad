"""Task-independent categorical policy voting over compatible rate-engine clients."""
import time

import numpy as np


def vote(probabilities, mode='majority'):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or min(p.shape) < 2 or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError('expected normalized finite member-by-action probabilities')
    mean = p.mean(axis=0)
    actions = p.argmax(axis=1)
    if mode == 'mean':
        return mean, actions
    if mode != 'majority':
        raise ValueError('mode must be majority or mean')
    counts = np.bincount(actions, minlength=p.shape[1])
    tied = np.flatnonzero(counts == counts.max())
    # All-different votes use average confidence among tied actions; exact ties
    # use lowest action index. No environment-specific safety rule is hidden here.
    action = int(tied[np.argmax(mean[tied])])
    result = np.zeros(p.shape[1])
    result[action] = 1
    return result, actions


class EnsembleClient:
    def __init__(self, members, mode='majority'):
        if len(members) < 2 or mode not in ('majority', 'mean'):
            raise ValueError('at least two members and a valid voting mode required')
        self.members = members
        self.ids = members[0].ids
        if any(member.ids != self.ids for member in members):
            raise ValueError('members must share ordered input ports and action semantics')
        self.mode = mode
        self.rows = []

    def infer(self, rates):
        started = time.perf_counter()
        probabilities = [member.infer(rates) for member in self.members]
        result, actions = vote(probabilities, self.mode)
        self.rows.append({'actions': actions.tolist(), 'selected': int(result.argmax()),
                          'latency_ms': (time.perf_counter() - started) * 1000})
        return result

    def statistics(self):
        if not self.rows:
            return {'decisions': 0}
        actions = np.array([r['actions'] for r in self.rows])
        latencies = np.array([r['latency_ms'] for r in self.rows])
        return {'decisions': len(actions),
                'unanimous_fraction': float(np.mean(np.all(actions == actions[:, :1], axis=1))),
                'all_distinct_fraction': float(np.mean([len(set(row)) == len(row) for row in actions])),
                'pairwise_disagreement': {f'{a}-{b}': float(np.mean(actions[:, a] != actions[:, b]))
                                          for a in range(len(self.members)) for b in range(a+1, len(self.members))},
                'latency_ms': {'mean': float(latencies.mean()), 'p95': float(np.percentile(latencies, 95)),
                               'max': float(latencies.max())},
                'latency_scope': 'sequential local HTTP member inference plus aggregation; excludes observation encoding and simulator stepping'}
